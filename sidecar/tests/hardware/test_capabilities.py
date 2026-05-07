from llm_chain_sidecar.hardware.capabilities import (
    Capability, capabilities_for_vram, MAX_PARAMS_BY_TIER,
)


def test_8gb_can_qlora_7b_but_not_full_ft_above_125m():
    caps = capabilities_for_vram(8.0)
    assert caps.qlora_max_params >= 7_000_000_000
    assert caps.lora_max_params <= 3_000_000_000
    assert caps.full_ft_max_params <= 125_000_000


def test_24gb_can_qlora_30b():
    caps = capabilities_for_vram(24.0)
    assert caps.qlora_max_params >= 13_000_000_000
    assert caps.full_ft_max_params >= 1_000_000_000


def test_128gb_unified_can_qlora_70b():
    caps = capabilities_for_vram(128.0, memory_kind="unified")
    assert caps.qlora_max_params >= 70_000_000_000


def test_pc_shared_memory_is_treated_as_zero():
    # PC "shared GPU memory" is slow PCIe DDR — must NOT count toward training capacity
    caps = capabilities_for_vram(64.0, memory_kind="shared")
    assert caps.qlora_max_params <= 1_000_000_000


def test_below_8gb_returns_minimal_tier():
    caps = capabilities_for_vram(4.0)
    assert caps.qlora_max_params < 7_000_000_000
