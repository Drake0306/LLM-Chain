from dataclasses import dataclass
from typing import Literal

# Tier table from design doc Section 4 (VRAM heuristics, 2026)
# Each entry: (min_vram_gb, qlora_max, lora_max, full_ft_max)
_TIERS = [
    (8.0,   7_000_000_000,  3_000_000_000,    125_000_000),
    (12.0,  13_000_000_000, 7_000_000_000,    350_000_000),
    (16.0,  13_000_000_000, 13_000_000_000,   500_000_000),
    (24.0,  34_000_000_000, 13_000_000_000,   1_300_000_000),
    (32.0,  30_000_000_000, 13_000_000_000,   1_500_000_000),
    (48.0,  70_000_000_000, 30_000_000_000,   3_000_000_000),
    (64.0,  70_000_000_000, 30_000_000_000,   3_000_000_000),
    (128.0, 70_000_000_000, 70_000_000_000,   7_000_000_000),
]

MAX_PARAMS_BY_TIER = _TIERS  # exported for the UI

MemoryKind = Literal["dedicated", "unified", "shared"]


@dataclass(frozen=True)
class Capability:
    qlora_max_params: int
    lora_max_params: int
    full_ft_max_params: int
    notes: str


def capabilities_for_vram(vram_gb: float, memory_kind: MemoryKind = "dedicated") -> Capability:
    """Return the max trainable params at each technique for a given VRAM tier.

    Apple Silicon "unified" memory really IS GPU-accessible at full speed — count it.
    PC "shared GPU memory" is system DDR over PCIe at ~30-60 GB/s, ~20x slower than
    real VRAM. The NVIDIA Windows driver silently spills there and tanks throughput
    10-50x rather than OOMing. Treat shared memory as effectively zero for training.
    """
    if memory_kind == "shared":
        # PC shared memory: ignore the tier table entirely, give a deliberately tiny budget
        return Capability(
            qlora_max_params=1_000_000_000,
            lora_max_params=350_000_000,
            full_ft_max_params=50_000_000,
            notes="Shared GPU memory (PCIe DDR) — slow; treated as minimal capacity.",
        )

    if vram_gb < _TIERS[0][0]:
        return Capability(
            qlora_max_params=1_000_000_000,
            lora_max_params=350_000_000,
            full_ft_max_params=50_000_000,
            notes="Below 8 GB — only very small models / tiny LoRAs.",
        )

    chosen = _TIERS[0]
    for tier in _TIERS:
        if vram_gb >= tier[0]:
            chosen = tier

    note = "Apple unified memory — ~75% addressable for GPU." if memory_kind == "unified" else ""
    return Capability(
        qlora_max_params=chosen[1],
        lora_max_params=chosen[2],
        full_ft_max_params=chosen[3],
        notes=note,
    )
