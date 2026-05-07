import threading
from collections.abc import Iterator
from pathlib import Path

from llm_chain_sidecar.trainers import make_trainer
from llm_chain_sidecar.trainers.base import EventType, TrainingEvent

from .store import RunStore
from .types import RunStatus

EVENTS_FILE_NAME = "events.jsonl"


def _events_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / EVENTS_FILE_NAME


def _append_event(run_dir: str | Path, event: TrainingEvent) -> None:
    """Append a single event as JSONL. Cheap to write, easy to replay later
    so the UI can show the full loss curve and log history when the user
    revisits a finished run."""
    line = event.model_dump_json() + "\n"
    with _events_path(run_dir).open("a") as f:
        f.write(line)


def read_events(run_dir: str | Path) -> list[dict]:
    p = _events_path(run_dir)
    if not p.exists():
        return []
    out: list[dict] = []
    import json
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # skip malformed line; keep replay best-effort
    return out


class RunExecutor:
    def __init__(self, store: RunStore) -> None:
        self.store = store
        # in-flight cancellation events keyed by run_id; populated only while
        # execute() is iterating
        self._cancel_events: dict[str, threading.Event] = {}

    def cancel(self, run_id: str) -> bool:
        ev = self._cancel_events.get(run_id)
        if ev is None:
            return False
        ev.set()
        return True

    def execute(self, run_id: str) -> Iterator[TrainingEvent]:
        run = self.store.get(run_id)
        # Browser EventSource auto-reconnects when the SSE connection drops
        # (network blip, sleep). Without these guards a reconnect would call
        # execute() a second time and either re-stream a finished run or fork
        # a duplicate trainer alongside the one already in flight.
        if run.status in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELED):
            return
        if run_id in self._cancel_events:
            return
        cancel_event = threading.Event()
        self._cancel_events[run_id] = cancel_event
        self.store.update_status(run_id, RunStatus.RUNNING)
        trainer = make_trainer(
            run.config.backend, run.config, run.output_dir, cancel_event=cancel_event
        )
        had_error = False
        try:
            try:
                for ev in trainer.train():
                    _append_event(run.output_dir, ev)
                    if ev.type == EventType.ERROR:
                        had_error = True
                        self.store.update_status(run_id, RunStatus.FAILED, error=ev.message)
                    yield ev
            except Exception as e:
                err_event = TrainingEvent(type=EventType.ERROR, message=str(e))
                _append_event(run.output_dir, err_event)
                self.store.update_status(run_id, RunStatus.FAILED, error=str(e))
                yield err_event
                return
            if cancel_event.is_set():
                self.store.update_status(run_id, RunStatus.CANCELED)
            elif not had_error:
                self.store.update_status(run_id, RunStatus.SUCCEEDED)
        finally:
            self._cancel_events.pop(run_id, None)
