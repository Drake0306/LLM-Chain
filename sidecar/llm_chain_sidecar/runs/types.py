import time
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class RunConfig(BaseModel):
    model_id: str
    backend: str            # "cuda", "mlx", etc.
    technique: str          # "lora", "qlora"
    dataset_path: str
    dataset_format: str = "jsonl_chat"
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32


class Run(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Monotonic, ns-resolution tiebreaker for created_at on platforms (Windows)
    # whose wall clock has ~15 ms resolution. Sorting uses (created_at, created_seq)
    # so back-to-back creates within a single clock tick still order deterministically.
    created_seq: int = Field(default_factory=time.monotonic_ns)
    status: RunStatus = RunStatus.PENDING
    config: RunConfig
    error: str | None = None
    output_dir: str | None = None
