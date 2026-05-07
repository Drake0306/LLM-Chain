from enum import Enum
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
    vram_gb: float          # 0 for CPU; for Apple Silicon = unified memory pool
    is_unified_memory: bool # True only for Apple Silicon
    driver_version: str | None = None


class HardwareReport(BaseModel):
    os: str                 # "Darwin", "Windows", "Linux"
    os_version: str
    cpu: CpuInfo
    system_ram_gb: float
    devices: list[GpuDevice]
