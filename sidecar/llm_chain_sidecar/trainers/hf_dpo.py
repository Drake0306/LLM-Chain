"""Direct preference optimisation (DPO) via TRL on HF backends (F-C10).

Mirrors the structure of HfCudaTrainer but swaps in trl.DPOTrainer
and the (prompt, chosen, rejected) dataset shape. Same outer shell
(HfStyleTrainer.train) handles START / event-translation / cancel /
DONE; only ``_run_training_loop`` differs.

MLX backends are explicitly out of scope — mlx_lm doesn't ship a DPO
trainer. The route layer rejects ``training_method=dpo`` with
``backend in {mlx, mlx_vlm}`` so this module is never reached for
MLX runs.

PEFT path: same as SFT — wrap the base model with LoraConfig +
get_peft_model, and DPOTrainer treats the wrapped model as the
policy network. The reference model is constructed automatically by
TRL when not passed (it deep-copies the base before the LoRA wrap).
"""
from __future__ import annotations

from collections.abc import Iterator

from ._hf_base import HfStyleTrainer


class HfDpoTrainer(HfStyleTrainer):
    """HF DPO trainer that runs on cuda / cpu / rocm devices.

    Uses the existing template-method shell — yields raw event dicts
    from a queue fed by a TRL TrainerCallback, just like the SFT
    flow. The only difference is the trl.DPOTrainer + DPOConfig
    construction + the (prompt, chosen, rejected) dataset shape.
    """

    def _run_training_loop(self) -> Iterator[dict]:  # pragma: no cover
        """Real implementation. Patched out in tests.

        Wires trl.DPOTrainer + peft.LoraConfig + the same callback
        bridge HfCudaTrainer uses. Loads the dataset via the JSONL_DPO
        loader (route validation already enforced format=jsonl_dpo)
        and converts to the column shape DPOTrainer expects.
        """
        import queue

        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
            run_in_background_with_sentinel,
        )
        from .hf_progress import emit_hf_download_progress

        # TRL >= 0.8 ships DPOTrainer; the API moved fields between
        # 0.8 and 0.9 (DPOConfig vs kwargs). Try the modern API first
        # then fall back so a frozen 0.8 install still trains.
        try:
            from trl import DPOConfig, DPOTrainer  # type: ignore
            _has_dpo_config = True
        except ImportError:
            from trl import DPOTrainer  # type: ignore
            DPOConfig = None  # type: ignore[assignment]
            _has_dpo_config = False

        ds_format = DatasetFormat(self.config.dataset_format)
        rows = ds_load(
            make_source(
                ds_format, self.config.dataset_path, self.config.text_column,
            )
        )

        events: queue.Queue[dict | None] = queue.Queue()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        with emit_hf_download_progress(events):
            tok = AutoTokenizer.from_pretrained(self.config.model_id)
            vocab_grew = ensure_pad_token(tok)
            base = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            ).to(device)
            if vocab_grew:
                base.resize_token_embeddings(len(tok))

        while not events.empty():
            yield events.get_nowait()

        # DPOTrainer accepts a HF Dataset with columns prompt / chosen
        # / rejected. The loader normalised these to plain strings so
        # we just construct the Dataset directly.
        ds = Dataset.from_list(rows)

        # PEFT wrap. Resume support mirrors hf_cuda — same adapter
        # shape applies because LoRA is orthogonal to the loss.
        resume_dir = resume_adapter_dir(
            self.output_dir, getattr(self.config, "resume_from", None),
        )
        if resume_dir is not None:
            model = PeftModel.from_pretrained(base, resume_dir, is_trainable=True)
        else:
            peft_cfg = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(base, peft_cfg)

        beta = getattr(self.config, "dpo_beta", 0.1)
        common_kwargs = dict(
            output_dir=self.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
        )
        if self.config.max_steps is not None:
            common_kwargs["max_steps"] = self.config.max_steps

        if _has_dpo_config:
            args = DPOConfig(beta=beta, **common_kwargs)  # type: ignore[arg-type]
            trainer = DPOTrainer(
                model=model,
                args=args,
                train_dataset=ds,
                tokenizer=tok,
                callbacks=[make_event_callback(events, self.cancel_event)],
            )
        else:
            # 0.8-style: DPOTrainer takes beta as a kwarg and uses
            # plain TrainingArguments under the hood.
            from transformers import TrainingArguments

            args = TrainingArguments(**common_kwargs)
            trainer = DPOTrainer(
                model=model,
                args=args,
                beta=beta,
                train_dataset=ds,
                tokenizer=tok,
                callbacks=[make_event_callback(events, self.cancel_event)],
            )

        run_in_background_with_sentinel(trainer.train, events)
        yield from pump_queue_until_sentinel(events)
