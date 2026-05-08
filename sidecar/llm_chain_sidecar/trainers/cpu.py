"""LoRA fine-tuning on CPU.

Mirrors HfCudaTrainer but pins everything to ``torch.device("cpu")`` and uses
float32 (bf16 isn't reliable on every CPU vendor). No bnb/QLoRA — quantized
training paths require CUDA. Hard cap on model size lives in capabilities.py
(``CPU_MAX_PARAMS``); the UI gates picker entries before they ever get here.
"""
from collections.abc import Iterator

from ._hf_base import HfStyleTrainer


class CpuTrainer(HfStyleTrainer):
    def _start_message(self) -> str:
        return f"Loading {self.config.model_id} on CPU"

    def _run_training_loop(self) -> Iterator[dict]:
        """Real training loop. Patched out in unit tests."""
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

        from llm_chain_sidecar.datasets import DatasetFormat, load_dataset as ds_load, make_source

        from ._text import (
            ensure_pad_token,
            make_event_callback,
            pump_queue_until_sentinel,
            resume_adapter_dir,
            row_to_text,
            run_in_background_with_sentinel,
        )
        from .hf_progress import emit_hf_download_progress

        ds_format = DatasetFormat(self.config.dataset_format)
        rows = ds_load(
            make_source(ds_format, self.config.dataset_path, self.config.text_column)
        )

        events: queue.Queue[dict | None] = queue.Queue()

        with emit_hf_download_progress(events):
            tok = AutoTokenizer.from_pretrained(self.config.model_id)
            vocab_grew = ensure_pad_token(tok)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, torch_dtype=torch.float32
            ).to("cpu")
            if vocab_grew:
                # See HfCudaTrainer for the rationale — the embedding matrix
                # has to grow to cover the [PAD] token ensure_pad_token just
                # added, or the forward pass crashes on index OOB.
                model.resize_token_embeddings(len(tok))
        while not events.empty():
            yield events.get_nowait()

        ds = Dataset.from_list(
            [{"text": row_to_text(r, ds_format, tok, self.config.text_column)} for r in rows]
        )
        ds = ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=512, padding="max_length"),
            remove_columns=["text"],
        )
        ds = ds.map(lambda b: {"labels": b["input_ids"]})

        # Resume support: see HfCudaTrainer for the rationale.
        resume_dir = resume_adapter_dir(self.output_dir, getattr(self.config, "resume_from", None))
        if resume_dir is not None:
            model = PeftModel.from_pretrained(model, resume_dir, is_trainable=True)
        else:
            peft_cfg = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_cfg)

        # See HfCudaTrainer for max_steps rationale.
        args_kwargs = dict(
            output_dir=self.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
            use_cpu=True,
        )
        if self.config.max_steps is not None:
            args_kwargs["max_steps"] = self.config.max_steps
        args = TrainingArguments(**args_kwargs)
        hf = HFTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            callbacks=[make_event_callback(events, self.cancel_event)],
        )

        run_in_background_with_sentinel(hf.train, events)
        yield from pump_queue_until_sentinel(events)
