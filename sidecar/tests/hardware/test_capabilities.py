from llm_chain_sidecar.hardware.capabilities import (
    CPU_MAX_PARAMS,
    Capability,
    capabilities_for_amd_vram,
    capabilities_for_cpu,
    capabilities_for_vram,
    MAX_PARAMS_BY_TIER,
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


def test_tier_boundary_at_12gb_is_inclusive():
    # 12.0 GB hits the 12 GB tier, not the 8 GB tier
    just_below = capabilities_for_vram(11.99)
    at = capabilities_for_vram(12.0)
    assert at.qlora_max_params >= just_below.qlora_max_params
    assert at.lora_max_params >= just_below.lora_max_params


def test_unified_memory_loses_25_percent_relative_to_dedicated():
    # 16 GB unified should be effectively 12 GB and gate to a lower tier than 16 GB dedicated
    dedicated = capabilities_for_vram(16.0, memory_kind="dedicated")
    unified = capabilities_for_vram(16.0, memory_kind="unified")
    assert unified.lora_max_params < dedicated.lora_max_params
    assert "unified_memory_overhead" in unified.warning_codes


def test_above_top_tier_saturates_to_128gb_row():
    # 192 GB Mac Studio Ultra shouldn't crash; should land at the top tier
    caps = capabilities_for_vram(192.0, memory_kind="unified")
    assert caps.qlora_max_params >= 70_000_000_000


def test_warning_codes_for_each_branch():
    assert capabilities_for_vram(64.0, memory_kind="shared").warning_codes == ("shared_memory_slow",)
    assert capabilities_for_vram(4.0).warning_codes == ("below_min_vram",)
    assert capabilities_for_vram(16.0, memory_kind="dedicated").warning_codes == ()


def test_cpu_capabilities_expose_a_nonzero_cpu_max():
    caps = capabilities_for_cpu()
    assert caps.cpu_max_params == CPU_MAX_PARAMS
    assert caps.cpu_max_params > 0
    # The GPU-tier numbers stay zero so nothing in the picker mistakenly thinks
    # CPU can run a 7B QLoRA.
    assert caps.qlora_max_params == 0
    assert caps.lora_max_params == 0
    assert caps.full_ft_max_params == 0
    assert "cpu_only_slow" in caps.warning_codes


def test_gpu_caps_carry_zero_cpu_max():
    caps = capabilities_for_vram(16.0, memory_kind="dedicated")
    assert caps.cpu_max_params == 0


def test_amd_vram_mirrors_cuda_tier_table():
    # Same memory math, different warning posture — the AMD path is a stub
    # until someone validates a real run on Radeon/Instinct silicon.
    cuda = capabilities_for_vram(24.0, memory_kind="dedicated")
    amd = capabilities_for_amd_vram(24.0)
    assert amd.qlora_max_params == cuda.qlora_max_params
    assert amd.lora_max_params == cuda.lora_max_params
    assert amd.full_ft_max_params == cuda.full_ft_max_params


def test_amd_vram_always_carries_rocm_unverified():
    for vram in (4.0, 8.0, 16.0, 24.0, 128.0):
        caps = capabilities_for_amd_vram(vram)
        assert "rocm_unverified" in caps.warning_codes, (
            f"AMD caps at {vram} GB lost the rocm_unverified warning"
        )


def test_amd_vram_notes_mention_experimental_status_and_issue_tracker():
    caps = capabilities_for_amd_vram(16.0)
    assert "experimental" in caps.notes.lower()
    assert "github.com/Drake0306/LLM-Chain/issues" in caps.notes


def test_amd_vram_below_min_still_marks_rocm_unverified():
    # When the underlying tier path tags below_min_vram, we keep both warnings
    caps = capabilities_for_amd_vram(4.0)
    assert "rocm_unverified" in caps.warning_codes
    assert "below_min_vram" in caps.warning_codes
