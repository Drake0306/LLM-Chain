import sys

from .base import EventType, Trainer, TrainingEvent
from .cpu import CpuTrainer
from .hf_cuda import HfCudaTrainer

__all__ = [
    "CpuTrainer",
    "EventType",
    "HfCudaTrainer",
    "Trainer",
    "TrainingEvent",
    "make_trainer",
]

if sys.platform == "darwin":
    from .mlx import MlxTrainer  # noqa: F401
    __all__.append("MlxTrainer")


def make_trainer(backend: str, *args, **kwargs) -> Trainer:
    if backend == "cuda":
        return HfCudaTrainer(*args, **kwargs)
    if backend == "cpu":
        return CpuTrainer(*args, **kwargs)
    if backend == "mlx":
        if sys.platform != "darwin":
            raise RuntimeError("MLX backend requires macOS")
        return MlxTrainer(*args, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")
