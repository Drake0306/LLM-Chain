from enum import Enum
from typing import Literal
from pydantic import BaseModel


class Backend(str, Enum):
    CUDA = "cuda"
    MPS = "mps"      # Apple Silicon via PyTorch MPS
    MLX = "mlx"      # Apple Silicon via MLX framework
    ROCM = "rocm"    # AMD (v1.1)
    XPU = "xpu"      # Intel (v1.1)
    CPU = "cpu"


class CpuInfo(BaseModel):
    cores: int
    name: str


class GpuDevice(BaseModel):
    backend: Backend
    name: str
    vram_gb: float                      # 0 for CPU; for unified-memory devices = unified pool
    memory_kind: Literal["dedicated", "unified", "shared"]
    driver_version: str | None = None
    # For unified-memory devices (Apple Silicon), the OS and other apps
    # already hold some of the unified pool. Probing this lets the
    # capability gate use what's *actually* available rather than the
    # theoretical hardware maximum, which prevents the "8 GB cap on a
    # 16 GB Mac with 8 GB used elsewhere → OOM during model load" foot-
    # gun. None for dedicated VRAM (GPU memory isn't shared) and CPU.
    available_vram_gb: float | None = None


class HardwareReport(BaseModel):
    os: str                 # "Darwin", "Windows", "Linux"
    os_version: str
    cpu: CpuInfo
    system_ram_gb: float
    devices: list[GpuDevice]
