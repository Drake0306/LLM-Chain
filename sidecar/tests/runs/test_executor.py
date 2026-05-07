import threading
from pathlib import Path
from unittest.mock import patch

from llm_chain_sidecar.runs.executor import RunExecutor, read_events
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


def test_executor_persists_events_for_replay(tmp_path: Path):
    """Each yielded event is appended to events.jsonl so the UI can replay
    the loss curve and log history when a user revisits the run later."""
    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    executor = RunExecutor(store)
    with patch("llm_chain_sidecar.runs.executor.make_trainer",
               return_value=FakeTrainer(cfg, run.output_dir)):
        list(executor.execute(run.id))

    replayed = read_events(run.output_dir)
    types = [e["type"] for e in replayed]
    assert types == ["start", "step", "step", "done"]
    losses = [e.get("loss") for e in replayed if e["type"] == "step"]
    assert losses == [2.0, 1.8]


def test_executor_persists_error_events(tmp_path: Path):
    """Exceptions inside trainer.train() generate an ERROR event that's
    persisted too, so a finished failure shows the full context on replay."""
    class BoomTrainer:
        def __init__(self, *_a, **_kw):
            pass

        def train(self):
            yield TrainingEvent(type=EventType.START)
            raise RuntimeError("disk full")

    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    executor = RunExecutor(store)

    with patch("llm_chain_sidecar.runs.executor.make_trainer", return_value=BoomTrainer()):
        events = list(executor.execute(run.id))

    assert events[-1].type == EventType.ERROR
    replayed = read_events(run.output_dir)
    assert replayed[-1]["type"] == "error"
    assert replayed[-1]["message"] == "disk full"


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


def test_executor_does_not_restart_terminal_run(tmp_path: Path):
    """If a browser EventSource auto-reconnects after a terminal run, calling
    execute() again must not start a second trainer."""
    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    store.update_status(run.id, RunStatus.SUCCEEDED)
    executor = RunExecutor(store)

    with patch("llm_chain_sidecar.runs.executor.make_trainer") as make:
        events = list(executor.execute(run.id))
    assert events == []
    assert make.call_count == 0
    assert store.get(run.id).status == RunStatus.SUCCEEDED


def test_executor_does_not_double_attach_to_running_run(tmp_path: Path):
    """A second concurrent execute() for the same run_id must not spawn a
    duplicate trainer."""
    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    executor = RunExecutor(store)

    blocker = threading.Event()

    class WaitingTrainer:
        def __init__(self, config, output_dir, cancel_event=None):
            pass

        def train(self):
            yield TrainingEvent(type=EventType.START)
            blocker.wait(timeout=1)
            yield TrainingEvent(type=EventType.DONE)

    def fake_make_trainer(_backend, config, output_dir, cancel_event=None):
        return WaitingTrainer(config, output_dir, cancel_event=cancel_event)

    with patch("llm_chain_sidecar.runs.executor.make_trainer", side_effect=fake_make_trainer):
        primary = executor.execute(run.id)
        next(primary)  # advance past START so cancel_event is registered
        # Second execute() while the first is still in flight: must yield
        # nothing and not call make_trainer again.
        with patch("llm_chain_sidecar.runs.executor.make_trainer") as second_make:
            second = list(executor.execute(run.id))
        assert second == []
        assert second_make.call_count == 0
        blocker.set()
        list(primary)  # drain
    assert store.get(run.id).status == RunStatus.SUCCEEDED


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
