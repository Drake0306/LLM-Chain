import itertools
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field  # noqa: F401  Field re-export

# Strictly-increasing per-process counter used as a tiebreaker on top of
# created_at. time.monotonic_ns() is only guaranteed non-decreasing, and on
# Windows runners we observed two back-to-back calls returning identical values
# — which made test_list_runs_returns_newest_first flaky. itertools.count is
# guaranteed to advance every call.
_seq_counter = itertools.count(1)


def _next_seq() -> int:
    return next(_seq_counter)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class RunConfig(BaseModel):
    # String fields carry max_length so a hand-crafted POST can't blow up
    # the run.json payload (and the disk usage that goes with it). The
    # bounds are generous — model ids and HF dataset ids are typically
    # well under 100 chars, paths well under 4 KB on every filesystem
    # we target — but tight enough that nothing reasonable hits them.
    model_id: str = Field(min_length=1, max_length=512)
    backend: str = Field(min_length=1, max_length=32)            # "cuda", "mlx", etc.
    technique: str = Field(min_length=1, max_length=16)          # "lora", "qlora"
    dataset_path: str = Field(max_length=4096)
    dataset_format: str = Field(default="jsonl_chat", max_length=32)
    text_column: str | None = Field(default=None, max_length=128)  # for CSV format
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32


class Run(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Strictly-increasing per-process tiebreaker for created_at. Windows wall
    # clock has ~15 ms resolution; sorting uses (created_at, created_seq) so
    # back-to-back creates inside a single clock tick still order deterministically.
    created_seq: int = Field(default_factory=_next_seq)
    status: RunStatus = RunStatus.PENDING
    config: RunConfig
    error: str | None = None
    output_dir: str | None = None
