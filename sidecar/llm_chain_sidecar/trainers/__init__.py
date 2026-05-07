import sys

from .base import EventType, Trainer, TrainingEvent
from .hf_cuda import HfCudaTrainer

__all__ = ["EventType", "HfCudaTrainer", "Trainer", "TrainingEvent", "make_trainer"]

if sys.platform == "darwin":
    from .mlx import MlxTrainer  # noqa: F401
    __all__.append("MlxTrainer")


def make_trainer(backend: str, *args, **kwargs) -> Trainer:
    if backend == "cuda":
        return HfCudaTrainer(*args, **kwargs)
    if backend == "mlx":
        if sys.platform != "darwin":
            raise RuntimeError("MLX backend requires macOS")
        return MlxTrainer(*args, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")
