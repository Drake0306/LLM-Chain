import threading
from pathlib import Path
from unittest.mock import patch

from llm_chain_sidecar.runs.executor import RunExecutor
from llm_chain_sidecar.runs.store import RunStore
from llm_chain_sidecar.runs.types import RunConfig, RunStatus
from llm_chain_sidecar.trainers.base import EventType, TrainingEvent


class FakeTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir

    def train(self):
        yield TrainingEvent(type=EventType.START)
        yield TrainingEvent(type=EventType.STEP, step=1, total_steps=2, loss=2.0)
        yield TrainingEvent(type=EventType.STEP, step=2, total_steps=2, loss=1.8)
        yield TrainingEvent(type=EventType.DONE)


def test_executor_runs_trainer_and_marks_succeeded(tmp_path: Path):
    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    executor = RunExecutor(store)
    with patch("llm_chain_sidecar.runs.executor.make_trainer",
               return_value=FakeTrainer(cfg, run.output_dir)):
        events = list(executor.execute(run.id))
    assert events[-1].type == EventType.DONE
    assert store.get(run.id).status == RunStatus.SUCCEEDED


def test_executor_marks_canceled_when_cancel_called(tmp_path: Path):
    """Cancellation flips the run to CANCELED instead of SUCCEEDED, even if
    the trainer ends with no error event."""

    captured: dict = {}

    class CancelAwareTrainer:
        def __init__(self, config, output_dir, cancel_event=None):
            self.cancel_event = cancel_event or threading.Event()
            captured["trainer"] = self

        def train(self):
            yield TrainingEvent(type=EventType.START)
            self.cancel_event.wait(timeout=1)
            yield TrainingEvent(type=EventType.CANCELED, message="Canceled by user")

    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    executor = RunExecutor(store)

    def fake_make_trainer(_backend, config, output_dir, cancel_event=None):
        return CancelAwareTrainer(config, output_dir, cancel_event=cancel_event)

    with patch("llm_chain_sidecar.runs.executor.make_trainer", side_effect=fake_make_trainer):
        gen = executor.execute(run.id)
        # consume START so the cancel_event has been registered
        first = next(gen)
        assert first.type == EventType.START
        assert executor.cancel(run.id) is True
        rest = list(gen)
    types = [e.type for e in rest]
    assert EventType.CANCELED in types
    assert store.get(run.id).status == RunStatus.CANCELED


def test_executor_cancel_returns_false_when_no_active_run(tmp_path: Path):
    store = RunStore(root=tmp_path)
    executor = RunExecutor(store)
    assert executor.cancel("not-a-real-run") is False


def test_executor_marks_failed_on_error_event(tmp_path: Path):
    class FailingTrainer:
        def __init__(self, config, output_dir):
            pass

        def train(self):
            yield TrainingEvent(type=EventType.START)
            yield TrainingEvent(type=EventType.ERROR, message="boom")

    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    executor = RunExecutor(store)
    with patch("llm_chain_sidecar.runs.executor.make_trainer",
               return_value=FailingTrainer(cfg, "")):
        list(executor.execute(run.id))
    saved = store.get(run.id)
    assert saved.status == RunStatus.FAILED
    assert "boom" in (saved.error or "")
