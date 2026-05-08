import importlib.util
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

# MLX trainer modules import cleanly even without the runtime mlx_lm / mlx_vlm
# packages installed (they shell out via subprocess). Module import failures
# would only happen if the trainer source itself broke; capture that
# separately for diagnostic purposes.
_mlx_module_import_error: Exception | None = None
if sys.platform == "darwin":
    try:
        from .mlx import MlxTrainer  # noqa: F401
        from .mlx_vlm import MlxVlmTrainer  # noqa: F401
        __all__.extend(["MlxTrainer", "MlxVlmTrainer"])
    except Exception as e:  # noqa: BLE001 — surfaced via make_trainer below
        _mlx_module_import_error = e


def _require_runtime_package(name: str, install_hint: str) -> None:
    """Pre-flight check that a runtime package is importable.

    The MLX trainers spawn ``python -m <pkg> lora`` as a subprocess, so a
    missing package only surfaces deep in the subprocess's traceback —
    e.g. a bare ``ModuleNotFoundError: No module named 'mlx_vlm'`` at the
    head of the captured tail. Checking importability before spawn lets
    us raise a single actionable RuntimeError that the route layer
    converts to a 400 with a concrete ``pip install ...`` hint.
    """
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(
            f"{name} is not installed; {install_hint}"
        )


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
        if _mlx_module_import_error is not None:
            raise RuntimeError(
                f"MLX trainer module failed to import: {_mlx_module_import_error}."
            )
        _require_runtime_package(
            "mlx_lm",
            "install it with `pip install mlx-lm` (or reinstall the macOS extra).",
        )
        return MlxTrainer(*args, **kwargs)
    if backend == "mlx_vlm":
        if sys.platform != "darwin":
            raise RuntimeError("MLX VLM backend requires macOS")
        if _mlx_module_import_error is not None:
            raise RuntimeError(
                f"MLX VLM trainer module failed to import: {_mlx_module_import_error}."
            )
        _require_runtime_package(
            "mlx_vlm",
            "install it with `pip install mlx-vlm`.",
        )
        return MlxVlmTrainer(*args, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")
