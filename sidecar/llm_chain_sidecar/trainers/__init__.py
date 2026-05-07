import sys

from .base import EventType, Trainer, TrainingEvent
from .cpu import CpuTrainer
from .hf_cuda import HfCudaTrainer
from .hf_rocm import HfRocmTrainer
from .hf_vlm import HfVlmTrainer

__all__ = [
    "CpuTrainer",
    "EventType",
    "HfCudaTrainer",
    "HfRocmTrainer",
    "HfVlmTrainer",
    "Trainer",
    "TrainingEvent",
    "make_trainer",
]

if sys.platform == "darwin":
    from .mlx import MlxTrainer  # noqa: F401
    from .mlx_vlm import MlxVlmTrainer  # noqa: F401
    __all__.extend(["MlxTrainer", "MlxVlmTrainer"])


def make_trainer(backend: str, *args, **kwargs) -> Trainer:
    if backend == "cuda":
        return HfCudaTrainer(*args, **kwargs)
    if backend == "cuda_vlm":
        return HfVlmTrainer(*args, **kwargs)
    if backend == "rocm":
        return HfRocmTrainer(*args, **kwargs)
    if backend == "cpu":
        return CpuTrainer(*args, **kwargs)
    if backend == "mlx":
        if sys.platform != "darwin":
            raise RuntimeError("MLX backend requires macOS")
        return MlxTrainer(*args, **kwargs)
    if backend == "mlx_vlm":
        if sys.platform != "darwin":
            raise RuntimeError("MLX VLM backend requires macOS")
        return MlxVlmTrainer(*args, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")
