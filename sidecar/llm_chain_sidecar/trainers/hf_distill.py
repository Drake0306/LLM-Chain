"""Knowledge distillation via inline teacher + student (F-C11).

The trainer loads the teacher (frozen, eval mode) alongside the student
(LoRA-wrapped, trainable) and computes a per-step loss that combines:

  - the standard cross-entropy term against the dataset labels
    (the SFT objective the student would have used on its own), and
  - a temperature-softened KL-divergence term between the student's
    logits and the teacher's logits

with weight ``α`` on the CE term and ``(1 - α)`` on the KL term. ``α =
0.5`` is the conventional starting point; ``α = 0`` is pure
distillation, ``α = 1`` collapses to plain SFT.

Why inline (both models in memory) rather than the two-phase "save
top-K teacher logits to disk" pipeline the HANDOFF described:
inline is simpler to ship and validate in one session, and the cost is
just GPU/CPU memory headroom for the teacher (which the user picks).
The two-phase variant remains a future optimisation — its docstring
hook is in this module's footer for whoever lands it.

Tokenizer compatibility: teacher and student MUST share a tokenizer
(same vocab, same token ids) — we softmax + KL across the logit
distributions, which is meaningless across mismatched vocabularies.
The trainer enforces this at load time and raises with a clear
message before the user wastes minutes of training on a broken
configuration.
"""
from __future__ import annotations

from collections.abc import Iterator

from ._hf_base import HfStyleTrainer


class HfDistillTrainer(HfStyleTrainer):
    """HF-backed distillation trainer for cuda / cpu / rocm.

    Same outer shell as HfCudaTrainer (handled by HfStyleTrainer.train);
    only ``_run_training_loop`` differs to load the teacher + wire the
    custom compute_loss.
    """

    def _start_message(self) -> str:
        teacher = getattr(self.config, "teacher_model_id", None)
        if teacher:
            return f"Loading {self.config.model_id} (teacher: {teacher})"
        return super()._start_message()

    def _run_training_loop(self) -> Iterator[dict]:  # pragma: no cover
        """Real implementation. Patched out in tests.

        Loads the teacher in eval mode, the student under PEFT, and
        runs HF Trainer with a Trainer subclass that overrides
        compute_loss to combine CE + KL.
        """
        import queue

        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer as HFTrainer,
            TrainingArguments,
        )

        from llm_chain_sidecar.datasets import (
            DatasetFormat,
            load_dataset as ds_load,
            make_source,
        )

        from ._text import (
            ensure_pad_token,
            make_event_callback,
            pump_queue_until_sentinel,
            resume_adapter_dir,
            row_to_text,
            run_in_background_with_sentinel,
        )
        from .hf_progress import emit_hf_download_progress

        teacher_id = getattr(self.config, "teacher_model_id", None)
        if not teacher_id:
            raise ValueError(
                "teacher_model_id is required for distillation training."
            )

        ds_format = DatasetFormat(self.config.dataset_format)
        rows = ds_load(
            make_source(
                ds_format, self.config.dataset_path, self.config.text_column,
            )
        )

        events: queue.Queue[dict | None] = queue.Queue()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        with emit_hf_download_progress(events):
            # Tokenizer comes from the student — the teacher MUST agree
            # on vocabulary (token ids) for KL across logits to be
            # meaningful. We verify by comparing vocab sizes after
            # loading both tokenizers; mismatches raise upfront.
            student_tok = AutoTokenizer.from_pretrained(self.config.model_id)
            teacher_tok = AutoTokenizer.from_pretrained(teacher_id)
            if student_tok.vocab_size != teacher_tok.vocab_size:
                raise ValueError(
                    f"Tokenizer vocab mismatch: student "
                    f"{self.config.model_id} has vocab_size "
                    f"{student_tok.vocab_size}, teacher {teacher_id} has "
                    f"{teacher_tok.vocab_size}. Distillation needs identical "
                    "vocabularies — pick a teacher from the same family."
                )
            vocab_grew = ensure_pad_token(student_tok)

            student = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, torch_dtype=dtype,
            ).to(device)
            if vocab_grew:
                student.resize_token_embeddings(len(student_tok))

            teacher = AutoModelForCausalLM.from_pretrained(
                teacher_id, torch_dtype=dtype,
            ).to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False

        while not events.empty():
            yield events.get_nowait()

        ds = Dataset.from_list(
            [
                {
                    "text": row_to_text(
                        r, ds_format, student_tok, self.config.text_column,
                    )
                }
                for r in rows
            ]
        )
        ds = ds.map(
            lambda b: student_tok(
                b["text"], truncation=True, max_length=512, padding="max_length",
            ),
            remove_columns=["text"],
        )
        ds = ds.map(lambda b: {"labels": b["input_ids"]})

        resume_dir = resume_adapter_dir(
            self.output_dir, getattr(self.config, "resume_from", None),
        )
        if resume_dir is not None:
            student = PeftModel.from_pretrained(student, resume_dir, is_trainable=True)
        else:
            peft_cfg = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            student = get_peft_model(student, peft_cfg)

        alpha = float(getattr(self.config, "distill_alpha", 0.5))
        T = float(getattr(self.config, "distill_temperature", 2.0))
        # Wrap HF Trainer so we can override compute_loss without
        # subclassing at module top level (keeps the heavy import
        # inside the lazy block).
        import torch.nn.functional as F

        class _DistillTrainer(HFTrainer):
            """HF Trainer subclass that adds a temperature-softened
            KL-divergence term against the frozen teacher's logits.

            CE: standard causal-LM cross-entropy from HFTrainer's
            super().compute_loss.
            KL: F.kl_div(log_softmax(student/T), softmax(teacher/T))
                * T**2 — the T**2 scale keeps the gradient magnitude
                comparable to the CE term across temperature choices,
                following Hinton et al. 2015.
            """

            def compute_loss(  # type: ignore[override]
                self, model, inputs, return_outputs=False, **kwargs,
            ):
                # Teacher forward — no_grad to skip the tape.
                with torch.no_grad():
                    teacher_out = teacher(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    )
                outputs = model(**inputs)
                ce_loss = outputs.loss
                student_logits = outputs.logits
                teacher_logits = teacher_out.logits
                # Shape parity check — vocab dim must match. If
                # tokenizers agreed but the models' lm_heads disagree
                # (rare but possible across pre-/post-RLHF variants),
                # surface here instead of NaN-ing silently.
                if student_logits.shape[-1] != teacher_logits.shape[-1]:
                    raise ValueError(
                        "Student and teacher logit dims disagree "
                        f"({student_logits.shape[-1]} vs "
                        f"{teacher_logits.shape[-1]}). Pick teacher / "
                        "student from the same family."
                    )
                kl = F.kl_div(
                    F.log_softmax(student_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                loss = alpha * ce_loss + (1.0 - alpha) * kl
                return (loss, outputs) if return_outputs else loss

        args_kwargs = dict(
            output_dir=self.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
        )
        if self.config.max_steps is not None:
            args_kwargs["max_steps"] = self.config.max_steps
        args = TrainingArguments(**args_kwargs)
        hf = _DistillTrainer(
            model=student,
            args=args,
            train_dataset=ds,
            callbacks=[make_event_callback(events, self.cancel_event)],
        )

        run_in_background_with_sentinel(hf.train, events)
        yield from pump_queue_until_sentinel(events)


# Future work: two-phase distillation (HANDOFF F-C11).
# - Phase 1: spawn the teacher over the dataset's prompts, capture
#   top-K logits per token, save to <run_dir>/teacher_logits.bin
# - Phase 2: load the saved logits, train the student with KL-div
#   against them (no teacher in memory during step 2)
# Memory: O(K × seq_len × N_rows × 4) on disk vs the current
# O(teacher params + student params) in RAM. Disk-store wins for
# big teachers + many epochs. Inline (this file) wins for one-shot
# distillations where the dataset is small.
