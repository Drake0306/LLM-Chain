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

# MLX trainers depend on mlx_lm / mlx_vlm being installed. On a Mac dev box
# they always are (pyproject pins them under the macos extra), but a user
# launching the sidecar from a packaged build with a missing wheel — or a
# bootstrap mid-install — would otherwise crash the whole sidecar at import
# time. Fall back to leaving them unregistered; make_trainer raises a clear
# error if the user actually picks the missing backend.
_mlx_import_error: Exception | None = None
if sys.platform == "darwin":
    try:
        from .mlx import MlxTrainer  # noqa: F401
        from .mlx_vlm import MlxVlmTrainer  # noqa: F401
        __all__.extend(["MlxTrainer", "MlxVlmTrainer"])
    except Exception as e:  # noqa: BLE001 — surfaced via make_trainer below
        _mlx_import_error = e


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
        if _mlx_import_error is not None:
            raise RuntimeError(
                f"MLX trainer unavailable: {_mlx_import_error}. "
                "Install mlx-lm with `pip install mlx-lm` (or reinstall the "
                "macOS extra)."
            )
        return MlxTrainer(*args, **kwargs)
    if backend == "mlx_vlm":
        if sys.platform != "darwin":
            raise RuntimeError("MLX VLM backend requires macOS")
        if _mlx_import_error is not None:
            raise RuntimeError(
                f"MLX VLM trainer unavailable: {_mlx_import_error}. "
                "Install mlx-vlm with `pip install mlx-vlm` (or reinstall the "
                "macOS extra)."
            )
        return MlxVlmTrainer(*args, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")
