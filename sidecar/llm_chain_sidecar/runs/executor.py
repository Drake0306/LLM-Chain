from collections.abc import Iterator

from llm_chain_sidecar.trainers import make_trainer
from llm_chain_sidecar.trainers.base import EventType, TrainingEvent

from .store import RunStore
from .types import RunStatus


class RunExecutor:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    def execute(self, run_id: str) -> Iterator[TrainingEvent]:
        run = self.store.get(run_id)
        self.store.update_status(run_id, RunStatus.RUNNING)
        trainer = make_trainer(run.config.backend, run.config, run.output_dir)
        had_error = False
        try:
            for ev in trainer.train():
                if ev.type == EventType.ERROR:
                    had_error = True
                    self.store.update_status(run_id, RunStatus.FAILED, error=ev.message)
                yield ev
        except Exception as e:
            self.store.update_status(run_id, RunStatus.FAILED, error=str(e))
            yield TrainingEvent(type=EventType.ERROR, message=str(e))
            return
        if not had_error:
            self.store.update_status(run_id, RunStatus.SUCCEEDED)
