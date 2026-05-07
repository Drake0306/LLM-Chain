from collections.abc import Iterator

from .base import EventType, Trainer, TrainingEvent


class HfCudaTrainer(Trainer):
    """LoRA fine-tuning via Hugging Face transformers + peft on CUDA.

    The actual HF Trainer doesn't natively yield events, so we use a
    TrainerCallback that pushes onto a queue and bridge that to a generator.
    For unit tests we patch _run_training_loop to inject fake events.
    """

    def train(self) -> Iterator[TrainingEvent]:
        yield TrainingEvent(type=EventType.START, message=f"Loading {self.config.model_id}")
        try:
            for raw in self._run_training_loop():
                kind = raw.get("type", "step")
                if kind == "download":
                    yield TrainingEvent(
                        type=EventType.DOWNLOAD,
                        bytes_done=raw.get("bytes_done"),
                        bytes_total=raw.get("bytes_total"),
                        message=raw.get("desc") or None,
                    )
                else:
                    yield TrainingEvent(
                        type=EventType.STEP,
                        step=raw["step"],
                        total_steps=raw["total_steps"],
                        loss=raw.get("loss"),
                        lr=raw.get("lr"),
                    )
        except Exception as e:
            yield TrainingEvent(type=EventType.ERROR, message=str(e))
            return
        if self.is_canceled():
            yield TrainingEvent(type=EventType.CANCELED, message="Canceled by user")
            return
        yield TrainingEvent(type=EventType.DONE, message=f"Saved to {self.output_dir}")

    def _run_training_loop(self) -> Iterator[dict]:
        """Real implementation. Patched out in tests.

        Wires HF Trainer + peft.LoraConfig + a callback that pushes events
        from a background thread onto a queue we consume here. The same
        queue carries download progress events emitted by hf_progress.
        """
        import queue
        from threading import Thread

        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer as HFTrainer,
            TrainerCallback,
            TrainingArguments,
        )

        from llm_chain_sidecar.datasets.loader import load_dataset as ds_load
        from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource

        from .hf_progress import emit_hf_download_progress

        ds_format = DatasetFormat(self.config.dataset_format)
        rows = ds_load(
            DatasetSource(
                format=ds_format,
                path=self.config.dataset_path,
                text_column=self.config.text_column,
            )
        )

        events: queue.Queue[dict | None] = queue.Queue()

        # Capture HF download bars while loading the tokenizer + model. Once
        # weights are local the context exits and tqdm goes back to normal,
        # so HF Trainer's own progress bar (which we don't want to broadcast)
        # is unaffected.
        with emit_hf_download_progress(events):
            tok = AutoTokenizer.from_pretrained(self.config.model_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, torch_dtype=torch.bfloat16
            ).to("cuda")
        # Drain any download events queued during model load before training
        # starts, so the consumer sees them in order.
        while not events.empty():
            yield events.get_nowait()

        def to_text(row):
            if ds_format in (DatasetFormat.JSONL_CHAT, DatasetFormat.JSONL_CHAT_VISION):
                return {"text": "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])}
            col = self.config.text_column or "text"
            return {"text": row[col]}

        ds = Dataset.from_list([to_text(r) for r in rows])
        ds = ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=512, padding="max_length"),
            remove_columns=["text"],
        )
        ds = ds.map(lambda b: {"labels": b["input_ids"]})

        peft_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

        cancel_event = self.cancel_event

        class Cb(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    events.put({
                        "type": "step",
                        "step": state.global_step,
                        "total_steps": state.max_steps,
                        "loss": logs["loss"],
                        "lr": logs.get("learning_rate"),
                    })

            def on_step_end(self, args, state, control, **kw):
                if cancel_event.is_set():
                    control.should_training_stop = True

            def on_train_end(self, args, state, control, **kw):
                events.put(None)

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
        )
        hf = HFTrainer(model=model, args=args, train_dataset=ds, callbacks=[Cb()])

        Thread(target=hf.train, daemon=True).start()
        while True:
            ev = events.get()
            if ev is None:
                break
            yield ev
