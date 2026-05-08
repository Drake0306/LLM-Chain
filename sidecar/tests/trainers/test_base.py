import pytest

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers.base import EventType, Trainer, TrainingEvent


def test_trainer_is_abstract():
    with pytest.raises(TypeError):
        Trainer(  # type: ignore[abstract]
            RunConfig(model_id="m", backend="cpu", technique="lora",
                      dataset_path="/tmp/x", epochs=1),
            output_dir="/tmp/out",
        )


def test_event_construction():
    e = TrainingEvent(type=EventType.STEP, step=1, total_steps=10,
                      loss=2.3, lr=2e-4)
    assert e.step == 1
    assert e.loss == 2.3


def test_make_trainer_surfaces_mlx_import_error_without_crashing_module(monkeypatch):
    """An MLX wheel that fails to import (mid-bootstrap, ABI skew) used to
    crash trainers/__init__ at sidecar startup, taking down /api/hardware
    too. Now the failure is captured and surfaced only when the user
    actually picks the mlx backend — non-MLX paths keep working."""
    import sys

    from llm_chain_sidecar import trainers as trainers_mod
    from llm_chain_sidecar.runs.types import RunConfig
    from llm_chain_sidecar.trainers import make_trainer

    if sys.platform != "darwin":
        pytest.skip("MLX is darwin-only; the import-fault path is darwin-specific.")

    cfg = RunConfig(model_id="m", backend="mlx", technique="lora",
                    dataset_path="/tmp/x", epochs=1)

    monkeypatch.setattr(trainers_mod, "_mlx_import_error", ImportError("mlx_lm broken"))
    with pytest.raises(RuntimeError, match="MLX trainer unavailable"):
        make_trainer("mlx", cfg, "/tmp/out")
    with pytest.raises(RuntimeError, match="MLX VLM trainer unavailable"):
        make_trainer("mlx_vlm", cfg, "/tmp/out")

    # Non-MLX backends still work.
    from llm_chain_sidecar.trainers import CpuTrainer
    cpu_cfg = RunConfig(model_id="m", backend="cpu", technique="lora",
                        dataset_path="/tmp/x", epochs=1)
    assert isinstance(make_trainer("cpu", cpu_cfg, "/tmp/out"), CpuTrainer)
