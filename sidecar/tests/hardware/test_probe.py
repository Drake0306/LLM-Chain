import platform
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from llm_chain_sidecar.hardware import probe as probe_mod
from llm_chain_sidecar.hardware.probe import probe_hardware
from llm_chain_sidecar.hardware.types import HardwareReport, Backend


def test_probe_returns_report_with_at_least_cpu():
    report = probe_hardware()
    assert isinstance(report, HardwareReport)
    assert report.os in ("Darwin", "Windows", "Linux")
    assert report.cpu.cores >= 1
    assert report.system_ram_gb > 0
    # CPU is always present as a fallback backend
    assert any(d.backend == Backend.CPU for d in report.devices)


def test_probe_is_idempotent():
    a = probe_hardware()
    b = probe_hardware()
    assert a.os == b.os
    assert a.cpu.cores == b.cpu.cores
    assert len(a.devices) == len(b.devices)


def test_apple_silicon_lists_mlx_only_no_mps():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        import pytest
        pytest.skip("Apple Silicon only")
    report = probe_hardware()
    backends = {d.backend for d in report.devices}
    assert Backend.MLX in backends
    assert Backend.MPS not in backends


def test_devices_have_memory_kind_field():
    report = probe_hardware()
    for d in report.devices:
        assert d.memory_kind in ("dedicated", "unified", "shared")


def _fake_torch_with_hip(hip_version: str | None, devices: list[tuple[str, int]]):
    """Build a fake torch module that looks like a ROCm or CUDA build.

    devices is a list of (name, total_memory_bytes) tuples.
    """
    fake_cuda = MagicMock()
    fake_cuda.is_available.return_value = bool(devices)
    fake_cuda.device_count.return_value = len(devices)

    def _props(i):
        name, mem = devices[i]
        return SimpleNamespace(name=name, total_memory=mem)

    fake_cuda.get_device_properties.side_effect = _props
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        version=SimpleNamespace(cuda=None if hip_version else "12.4", hip=hip_version),
    )
    return fake_torch


def test_rocm_probe_detects_hip_version():
    fake = _fake_torch_with_hip("6.0.32830-d62f6a171", [("AMD Radeon RX 7900 XTX", 24 * 1024**3)])
    with patch.dict("sys.modules", {"torch": fake}):
        devices = probe_mod._probe_rocm()
    assert len(devices) == 1
    assert devices[0].backend == Backend.ROCM
    assert "Radeon" in devices[0].name
    assert devices[0].vram_gb == 24.0
    assert devices[0].memory_kind == "dedicated"
    assert devices[0].driver_version == "6.0.32830-d62f6a171"


def test_rocm_probe_returns_empty_when_no_hip():
    fake = _fake_torch_with_hip(None, [("NVIDIA GeForce RTX 4090", 24 * 1024**3)])
    with patch.dict("sys.modules", {"torch": fake}):
        assert probe_mod._probe_rocm() == []


def test_cuda_probe_skips_rocm_builds():
    # On a ROCm-built PyTorch, torch.cuda.is_available() is True but
    # torch.version.hip is set — we must NOT mislabel that as a CUDA device.
    fake = _fake_torch_with_hip("6.0.0", [("AMD Radeon RX 7900 XTX", 24 * 1024**3)])
    with patch.dict("sys.modules", {"torch": fake}):
        assert probe_mod._probe_cuda() == []


def test_rocm_probe_handles_missing_torch():
    # Simulate torch being absent (the ImportError branch)
    with patch.dict("sys.modules", {"torch": None}):
        # ``import torch`` with sys.modules[torch] = None raises ImportError
        # — that's exactly what we want to exercise.
        assert probe_mod._probe_rocm() == []
