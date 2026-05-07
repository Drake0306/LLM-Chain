import logging
import platform
import psutil
from .types import Backend, CpuInfo, GpuDevice, HardwareReport

log = logging.getLogger(__name__)


def probe_hardware() -> HardwareReport:
    devices: list[GpuDevice] = []
    devices.extend(_probe_cuda())
    devices.extend(_probe_apple())
    devices.append(GpuDevice(
        backend=Backend.CPU, name="CPU",
        vram_gb=0.0, memory_kind="dedicated",
    ))

    return HardwareReport(
        os=platform.system(),
        os_version=platform.release(),
        cpu=CpuInfo(cores=psutil.cpu_count(logical=False) or 1, name=_cpu_name()),
        system_ram_gb=round(psutil.virtual_memory().total / (1024**3), 2),
        devices=devices,
    )


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            return f"Apple Silicon ({platform.machine()})"
    return platform.processor() or platform.machine() or "unknown"


def _probe_cuda() -> list[GpuDevice]:
    try:
        import torch
    except ImportError:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            out.append(GpuDevice(
                backend=Backend.CUDA,
                name=props.name,
                vram_gb=round(props.total_memory / (1024**3), 2),
                memory_kind="dedicated",
                driver_version=getattr(torch.version, "cuda", None),
            ))
        return out
    except Exception as e:
        log.warning("CUDA probe failed: %s", e, exc_info=True)
        return []


def _probe_apple() -> list[GpuDevice]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    return [GpuDevice(
        backend=Backend.MLX,
        name="Apple Silicon GPU (MLX)",
        vram_gb=ram_gb,
        memory_kind="unified",
    )]
