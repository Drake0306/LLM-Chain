"""LoRA fine-tuning via Hugging Face transformers + peft on CUDA.

The actual HF Trainer doesn't natively yield events, so we use a
TrainerCallback that pushes onto a queue and bridge that to a generator.
For unit tests we patch _run_training_loop to inject fake events.
"""
from collections.abc import Iterator

from ._hf_base import HfStyleTrainer


class HfCudaTrainer(HfStyleTrainer):
    def _run_training_loop(self) -> Iterator[dict]:
        """Real implementation. Patched out in tests.

        Wires HF Trainer + peft.LoraConfig + a callback that pushes events
        from a background thread onto a queue we consume here. The same
        queue carries download progress events emitted by hf_progress.
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

        # Capture HF download bars while loading the tokenizer + model. Once
        # weights are local the context exits and tqdm goes back to normal,
        # so HF Trainer's own progress bar (which we don't want to broadcast)
        # is unaffected.
        with emit_hf_download_progress(events):
            tok = AutoTokenizer.from_pretrained(self.config.model_id)
            vocab_grew = ensure_pad_token(tok)

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, torch_dtype=torch.bfloat16
            ).to("cuda")
            if vocab_grew:
                # ensure_pad_token added a brand-new special token; the model
                # we just loaded still has the original embedding rows, so any
                # input id pointing at the new token would crash with
                # CUDA-side index-out-of-bounds. Grow the embedding matrix to
                # match the tokenizer.
                model.resize_token_embeddings(len(tok))
        # Drain any download events queued during model load before training
        # starts, so the consumer sees them in order.
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

        # Resume support: if the parent run's adapter is on disk, load it
        # over the base model so training continues from those weights
        # instead of fresh random LoRA init. The parent's LoraConfig is
        # baked into adapter_config.json — the route enforces that the
        # current run's rank/alpha match so the shapes line up.
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

        # max_steps: see RunConfig docstring. HF's TrainingArguments
        # has its own max_steps param that takes precedence over
        # num_train_epochs when > 0; mirror our optional config.
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
        hf = HFTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            callbacks=[make_event_callback(events, self.cancel_event)],
        )

        run_in_background_with_sentinel(hf.train, events)
        yield from pump_queue_until_sentinel(events)
