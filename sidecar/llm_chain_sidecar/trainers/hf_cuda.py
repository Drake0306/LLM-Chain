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
        yield TrainingEvent(type=EventType.DONE, message=f"Saved to {self.output_dir}")

    def _run_training_loop(self) -> Iterator[dict]:
        """Real implementation. Patched out in tests.

        Wires HF Trainer + peft.LoraConfig + a callback that pushes events
        from a background thread onto a queue we consume here.
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

        rows = ds_load(DatasetSource(format=DatasetFormat.JSONL_CHAT, path=self.config.dataset_path))
        tok = AutoTokenizer.from_pretrained(self.config.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def to_text(row):
            return {"text": "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])}

        ds = Dataset.from_list([to_text(r) for r in rows])
        ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=512, padding="max_length"))

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, torch_dtype=torch.bfloat16
        ).to("cuda")
        peft_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

        events: queue.Queue[dict | None] = queue.Queue()

        class Cb(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    events.put({
                        "step": state.global_step,
                        "total_steps": state.max_steps,
                        "loss": logs["loss"],
                        "lr": logs.get("learning_rate"),
                    })

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
