import logging
import platform
import psutil
from .types import Backend, CpuInfo, GpuDevice, HardwareReport

log = logging.getLogger(__name__)


def probe_hardware() -> HardwareReport:
    devices: list[GpuDevice] = []
    devices.extend(_probe_cuda())
    devices.extend(_probe_rocm())
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
        # ROCm builds of PyTorch also expose torch.cuda.is_available() == True
        # but set torch.version.hip and leave torch.version.cuda as None. Hand
        # that case off to _probe_rocm so we don't mislabel an AMD GPU as NVIDIA.
        if getattr(torch.version, "hip", None):
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


def _probe_rocm() -> list[GpuDevice]:
    """Detect AMD GPUs via the ROCm-built PyTorch.

    ROCm reuses the torch.cuda namespace, so a ROCm install reports devices
    via torch.cuda.* but sets torch.version.hip instead of torch.version.cuda.
    We surface what we find, but the trainer + UI mark these devices as
    experimental — see HfRocmTrainer and the rocm_unverified warning code.
    """
    try:
        import torch
    except ImportError:
        return []
    try:
        hip_version = getattr(torch.version, "hip", None)
        if not hip_version:
            return []
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            out.append(GpuDevice(
                backend=Backend.ROCM,
                name=props.name,
                vram_gb=round(props.total_memory / (1024**3), 2),
                memory_kind="dedicated",
                driver_version=hip_version,
            ))
        return out
    except Exception as e:
        log.warning("ROCm probe failed: %s", e, exc_info=True)
        return []


def _probe_apple() -> list[GpuDevice]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    vm = psutil.virtual_memory()
    total_gb = round(vm.total / (1024**3), 2)
    # Apple unified memory: the GPU shares the system RAM pool with every
    # other process. ``vm.available`` is psutil's best estimate of "memory
    # that can be allocated without swapping", which is what's actually
    # usable for training right now. Reporting it lets the capability
    # gate cap by the smaller of (theoretical 75% of total) and
    # (actually-available), so a Mac with Chrome + IDE eating 8 GB
    # doesn't promise a 12 GB tier and OOM during model load.
    available_gb = round(vm.available / (1024**3), 2)
    return [GpuDevice(
        backend=Backend.MLX,
        name="Apple Silicon GPU (MLX)",
        vram_gb=total_gb,
        memory_kind="unified",
        available_vram_gb=available_gb,
    )]
