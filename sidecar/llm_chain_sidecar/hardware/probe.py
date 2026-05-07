import platform
import psutil
from .types import Backend, CpuInfo, GpuDevice, HardwareReport


def probe_hardware() -> HardwareReport:
    devices: list[GpuDevice] = []
    devices.extend(_probe_cuda())
    devices.extend(_probe_apple())
    devices.append(GpuDevice(
        backend=Backend.CPU, name="CPU",
        vram_gb=0.0, is_unified_memory=False,
    ))

    return HardwareReport(
        os=platform.system(),
        os_version=platform.release(),
        cpu=CpuInfo(cores=psutil.cpu_count(logical=False) or 1, name=platform.processor() or "unknown"),
        system_ram_gb=round(psutil.virtual_memory().total / (1024**3), 2),
        devices=devices,
    )


def _probe_cuda() -> list[GpuDevice]:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            out.append(GpuDevice(
                backend=Backend.CUDA,
                name=props.name,
                vram_gb=round(props.total_memory / (1024**3), 2),
                is_unified_memory=False,
                driver_version=getattr(torch.version, "cuda", None),
            ))
        return out
    except Exception:
        return []


def _probe_apple() -> list[GpuDevice]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    # Apple Silicon: unified memory == system RAM
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    devices = [GpuDevice(
        backend=Backend.MLX,
        name="Apple Silicon GPU (MLX)",
        vram_gb=ram_gb,
        is_unified_memory=True,
    )]
    try:
        import torch
        if torch.backends.mps.is_available():
            devices.append(GpuDevice(
                backend=Backend.MPS,
                name="Apple Silicon GPU (PyTorch MPS)",
                vram_gb=ram_gb,
                is_unified_memory=True,
            ))
    except Exception:
        pass
    return devices
