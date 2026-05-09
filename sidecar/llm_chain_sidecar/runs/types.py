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
    # When set, the trainer continues from this earlier run's adapter
    # instead of starting from random LoRA weights. The MLX path passes
    # ``--resume-adapter-file`` to mlx_lm; the HF/CPU path loads the
    # adapter via ``PeftModel.from_pretrained`` before training starts.
    # Validated at the route level: must reference an existing
    # SUCCEEDED run on the same device family (so the adapter shape
    # matches).
    resume_from: str | None = Field(default=None, max_length=64)
    # Hard cap on iterations — overrides the epoch-based default. Used
    # by the learning-rate finder to spawn 10-step mini-runs without
    # rewriting the trainer's iteration math. ``None`` means "use
    # epochs * 100" (MLX) or ``num_train_epochs`` (HF Trainer); a
    # positive int caps below that. Validated at the route boundary.
    max_steps: int | None = Field(default=None, ge=1)
    # Optional tag the system uses to group special-purpose runs.
    # ``"lr_finder"`` marks runs spawned by the LR finder so the UI
    # can hide them from the main Runs list and surface them only in
    # the finder's results view.
    purpose: str | None = Field(default=None, max_length=32)
    # F-C10: training method. ``"sft"`` (default) drives the existing
    # supervised-fine-tune pipelines via HF Trainer / mlx_lm;
    # ``"dpo"`` switches HF backends to TRL's DPOTrainer and requires
    # a ``jsonl_dpo`` dataset format. mlx_lm doesn't have DPO yet so
    # mlx backends reject training_method="dpo" upfront.
    training_method: str = Field(default="sft", max_length=16)
    # F-C10: KL-penalty weight for DPO. Default mirrors TRL's own
    # default; lower values make the policy track the reference model
    # more closely. Ignored when training_method != "dpo".
    dpo_beta: float = Field(default=0.1, gt=0, le=10)


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
