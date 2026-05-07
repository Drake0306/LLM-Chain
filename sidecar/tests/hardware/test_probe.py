import platform

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
