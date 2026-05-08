"""Contract tests for the HfStyleTrainer template-method base.

These pin the behavior the base class promises subclasses without
needing transformers / torch / a real training loop. The point is that
if someone touches the START/STEP/DOWNLOAD/ERROR/CANCELED/DONE shell or
the dict→TrainingEvent translation, a fast unit test catches the
regression — not a slow integration test that depends on a model
download.
"""
import threading
from collections.abc import Iterator

import pytest

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers._hf_base import HfStyleTrainer
from llm_chain_sidecar.trainers.base import EventType, TrainingEvent


class _ScriptedTrainer(HfStyleTrainer):
    """Trainer that yields whatever raw events the test feeds in."""

    def __init__(self, raw_events, *, output_dir="/tmp", cancel_event=None):
        cfg = RunConfig(
            model_id="m", backend="cpu", technique="lora",
            dataset_path="/tmp/x", epochs=1,
        )
        super().__init__(cfg, output_dir=output_dir, cancel_event=cancel_event)
        self._scripted = list(raw_events)

    def _run_training_loop(self) -> Iterator[dict]:
        for ev in self._scripted:
            if isinstance(ev, BaseException):
                raise ev
            yield ev


def test_train_emits_start_then_step_then_done():
    trainer = _ScriptedTrainer([
        {"step": 1, "total_steps": 2, "loss": 2.5, "lr": 2e-4},
        {"step": 2, "total_steps": 2, "loss": 1.9, "lr": 2e-4},
    ])
    events = list(trainer.train())
    types = [e.type for e in events]
    assert types == [EventType.START, EventType.STEP, EventType.STEP, EventType.DONE]
    assert events[1].loss == 2.5
    assert events[-1].message and events[-1].message.startswith("Saved to")


def test_train_translates_download_dict_into_download_event():
    trainer = _ScriptedTrainer([
        {"type": "download", "bytes_done": 100, "bytes_total": 1000, "desc": "model.safetensors"},
        {"step": 1, "total_steps": 1, "loss": 1.0},
    ])
    events = list(trainer.train())
    download = next(e for e in events if e.type == EventType.DOWNLOAD)
    assert download.bytes_done == 100
    assert download.bytes_total == 1000
    assert download.message == "model.safetensors"


def test_train_surfaces_loop_exception_as_single_error_event():
    """The base wraps the inner loop in try/except so a raised exception
    becomes one ERROR TrainingEvent rather than tearing through the
    SSE generator with a 500."""
    trainer = _ScriptedTrainer([
        {"step": 1, "total_steps": 2, "loss": 2.5},
        RuntimeError("CUDA out of memory"),
    ])
    events = list(trainer.train())
    assert events[-1].type == EventType.ERROR
    assert "CUDA out of memory" in (events[-1].message or "")
    # No DONE event after an ERROR — the trainer returns early.
    assert not any(e.type == EventType.DONE for e in events)


def test_train_emits_canceled_when_cancel_event_set_before_done():
    cancel = threading.Event()
    cancel.set()
    trainer = _ScriptedTrainer(
        [{"step": 1, "total_steps": 1, "loss": 1.0}], cancel_event=cancel
    )
    events = list(trainer.train())
    assert events[-1].type == EventType.CANCELED


def test_translate_step_dict_carries_lr_and_total_steps():
    """Public contract: STEP events expose total_steps so the UI can
    render a progress bar without polling the run config."""
    ev = HfStyleTrainer._translate(
        {"step": 7, "total_steps": 100, "loss": 1.23, "lr": 1e-4}
    )
    assert ev.type == EventType.STEP
    assert ev.step == 7
    assert ev.total_steps == 100
    assert ev.loss == pytest.approx(1.23)
    assert ev.lr == pytest.approx(1e-4)


def test_default_start_message_uses_model_id():
    trainer = _ScriptedTrainer([])
    # First event is START; default message is "Loading <model_id>".
    first = next(iter(trainer.train()))
    assert first.type == EventType.START
    assert "Loading m" in (first.message or "")
