"""LoRA fine-tuning for vision-language models on CUDA via HF + peft.

Same shape as HfCudaTrainer but loads ``AutoProcessor`` (image+text together)
and ``AutoModelForVision2Seq`` instead of a plain causal LM. peft LoRA targets
only the language-model linears (``q_proj``, ``k_proj``, ``v_proj``,
``o_proj``); the vision tower stays frozen because target_modules don't match
its parameter names. That's deliberate — fine-tuning the vision encoder needs
much more data than a hobby user has, and freezing it cuts memory + step time
roughly in half.
"""
from collections.abc import Iterator

from .base import EventType, Trainer, TrainingEvent


class HfVlmTrainer(Trainer):
    def train(self) -> Iterator[TrainingEvent]:
        yield TrainingEvent(
            type=EventType.START,
            message=f"Loading VLM {self.config.model_id} on CUDA",
        )
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
        """Real training loop. Patched out in unit tests.

        Loads the VLM processor + model, builds a peft LoRA config that only
        targets the LM linears, and runs HF Trainer. The dataset is normalized
        from our chat-vision rows into the processor's expected message format
        (a list of ``{"role", "content": [parts...]}`` per row).
        """
        import queue
        from threading import Thread

        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForVision2Seq,
            AutoProcessor,
            Trainer as HFTrainer,
            TrainerCallback,
            TrainingArguments,
        )

        from llm_chain_sidecar.datasets.loader import load_dataset as ds_load
        from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource

        from .hf_progress import emit_hf_download_progress

        rows = ds_load(
            DatasetSource(format=DatasetFormat.JSONL_CHAT_VISION, path=self.config.dataset_path)
        )

        events: queue.Queue[dict | None] = queue.Queue()

        with emit_hf_download_progress(events):
            processor = AutoProcessor.from_pretrained(self.config.model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_id, torch_dtype=torch.bfloat16
            ).to("cuda")
        while not events.empty():
            yield events.get_nowait()

        ds = Dataset.from_list(_rows_to_processor_inputs(rows, processor))

        peft_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            # Target only the LM attention projections — the vision tower is
            # frozen by virtue of its parameter names not matching these.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
            remove_unused_columns=False,
        )
        hf = HFTrainer(model=model, args=args, train_dataset=ds, callbacks=[Cb()])

        Thread(target=hf.train, daemon=True).start()
        while True:
            ev = events.get()
            if ev is None:
                break
            yield ev


def _rows_to_processor_inputs(rows: list[dict], processor) -> list[dict]:
    """Materialize processor inputs from chat-vision rows.

    The chat-vision loader hands us absolute image paths; we open each image
    here and let the processor build pixel_values + input_ids together. Done
    eagerly (not in a Dataset.map) because Qwen2-VL's processor returns
    nested tensors that don't pickle well across worker processes.
    """
    from PIL import Image

    out = []
    for row in rows:
        msgs = row["messages"]
        # Convert {"type": "image", "path": ...} → {"type": "image"} so the
        # chat template doesn't choke on our path field; we hand the actual
        # PIL image to the processor separately.
        template_msgs = []
        images = []
        for m in msgs:
            tmpl_parts = []
            for part in m["content"]:
                if part["type"] == "image":
                    images.append(Image.open(part["path"]).convert("RGB"))
                    tmpl_parts.append({"type": "image"})
                else:
                    tmpl_parts.append({"type": "text", "text": part["text"]})
            template_msgs.append({"role": m["role"], "content": tmpl_parts})
        text = processor.apply_chat_template(template_msgs, add_generation_prompt=False)
        inputs = processor(text=[text], images=images or None, return_tensors="pt", padding=True)
        # Flatten the single-row batch dim so the HF Dataset stores per-row
        # tensors. labels = input_ids for causal LM.
        flat = {k: v.squeeze(0) for k, v in inputs.items()}
        flat["labels"] = flat["input_ids"].clone()
        out.append(flat)
    return out
