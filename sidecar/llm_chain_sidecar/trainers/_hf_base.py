"""Template-method base class for HF-Trainer-based trainers.

The CPU, CUDA, and VLM trainers used to inline the same outer ``train()``
shell — START event, queue-driven event translation loop, exception →
ERROR conversion, cancel detection, DONE — three times verbatim.
``HfStyleTrainer`` owns that shell so each concrete subclass only has to
implement ``_run_training_loop``: a generator that yields ``download``
and ``step`` dicts. The base translates those dicts into the public
``TrainingEvent`` shape and handles every terminal-state branch.

This isn't a generic abstraction — it's specific to the HF/peft pipeline
where training events flow through a queue.Queue (see _text.py for the
helpers HF-style trainers use to drive that queue). The MLX subprocess
trainers don't fit this shape and have their own base class.
"""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator

from .base import EventType, Trainer, TrainingEvent


class HfStyleTrainer(Trainer):
    """Trainers that drive HF Trainer + peft via a queue inherit from this.

    Concrete subclasses implement:
      - ``_run_training_loop()`` — generator yielding raw event dicts:
        ``{"type": "download", "bytes_done": int, "bytes_total": int, "desc": str}``
        for download progress, or
        ``{"step": int, "total_steps": int, "loss": float, "lr": float}``
        for training steps. Anything else without a ``type`` key is
        treated as a step.
      - ``_start_message()`` — short user-facing string for the first
        START event. Defaults to "Loading <model_id>".
    """

    def train(self) -> Iterator[TrainingEvent]:
        yield TrainingEvent(type=EventType.START, message=self._start_message())
        try:
            for raw in self._run_training_loop():
                yield self._translate(raw)
        except Exception as e:  # noqa: BLE001 — surfaced as a single ERROR event
            yield TrainingEvent(type=EventType.ERROR, message=str(e))
            return
        if self.is_canceled():
            yield TrainingEvent(type=EventType.CANCELED, message="Canceled by user")
            return
        yield TrainingEvent(type=EventType.DONE, message=f"Saved to {self.output_dir}")

    @abstractmethod
    def _run_training_loop(self) -> Iterator[dict]:
        """Yield raw event dicts from the underlying HF training pipeline."""

    def _start_message(self) -> str:
        return f"Loading {self.config.model_id}"

    @staticmethod
    def _translate(raw: dict) -> TrainingEvent:
        """Map a raw event dict from ``_run_training_loop`` to a public event.

        Kept as a staticmethod so subclass tests that drive
        ``_run_training_loop`` directly can verify the dict shape without
        instantiating the trainer.
        """
        if raw.get("type") == "download":
            return TrainingEvent(
                type=EventType.DOWNLOAD,
                bytes_done=raw.get("bytes_done"),
                bytes_total=raw.get("bytes_total"),
                message=raw.get("desc") or None,
            )
        return TrainingEvent(
            type=EventType.STEP,
            step=raw["step"],
            total_steps=raw["total_steps"],
            loss=raw.get("loss"),
            lr=raw.get("lr"),
        )
