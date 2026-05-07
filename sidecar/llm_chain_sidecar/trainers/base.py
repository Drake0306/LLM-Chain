import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum

from pydantic import BaseModel

from llm_chain_sidecar.runs.types import RunConfig


class EventType(str, Enum):
    START = "start"
    STEP = "step"
    EPOCH_END = "epoch_end"
    DOWNLOAD = "download"
    LOG = "log"  # raw progress / phase line forwarded from a subprocess trainer
    DONE = "done"
    ERROR = "error"
    CANCELED = "canceled"


class TrainingEvent(BaseModel):
    type: EventType
    step: int = 0
    total_steps: int = 0
    epoch: int = 0
    loss: float | None = None
    lr: float | None = None
    message: str | None = None
    bytes_done: int | None = None
    bytes_total: int | None = None


class Trainer(ABC):
    def __init__(
        self,
        config: RunConfig,
        output_dir: str,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.cancel_event = cancel_event or threading.Event()

    def is_canceled(self) -> bool:
        return self.cancel_event.is_set()

    @abstractmethod
    def train(self) -> Iterator[TrainingEvent]:
        """Yield TrainingEvent updates as training progresses."""
