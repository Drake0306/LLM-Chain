from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum

from pydantic import BaseModel

from llm_chain_sidecar.runs.types import RunConfig


class EventType(str, Enum):
    START = "start"
    STEP = "step"
    EPOCH_END = "epoch_end"
    DONE = "done"
    ERROR = "error"


class TrainingEvent(BaseModel):
    type: EventType
    step: int = 0
    total_steps: int = 0
    epoch: int = 0
    loss: float | None = None
    lr: float | None = None
    message: str | None = None


class Trainer(ABC):
    def __init__(self, config: RunConfig, output_dir: str) -> None:
        self.config = config
        self.output_dir = output_dir

    @abstractmethod
    def train(self) -> Iterator[TrainingEvent]:
        """Yield TrainingEvent updates as training progresses."""
