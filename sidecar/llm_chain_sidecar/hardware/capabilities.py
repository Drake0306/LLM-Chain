from dataclasses import dataclass
from typing import Literal

# Tier table from design doc Section 4 (VRAM heuristics, 2026)
# Each entry: (min_vram_gb, qlora_max, lora_max, full_ft_max)
_TIERS = [
    (8.0,   7_000_000_000,  3_000_000_000,    125_000_000),
    (12.0,  13_000_000_000, 7_000_000_000,    350_000_000),
    (16.0,  13_000_000_000, 13_000_000_000,   500_000_000),
    (24.0,  34_000_000_000, 13_000_000_000,   1_300_000_000),
    (32.0,  34_000_000_000, 13_000_000_000,   1_500_000_000),
    (48.0,  70_000_000_000, 30_000_000_000,   3_000_000_000),
    (64.0,  70_000_000_000, 30_000_000_000,   3_000_000_000),
    (128.0, 70_000_000_000, 70_000_000_000,   7_000_000_000),
]

MAX_PARAMS_BY_TIER = _TIERS  # exported for the UI

MemoryKind = Literal["dedicated", "unified", "shared"]


# CPU fallback: hard cap on what we'll let users LoRA on a stock laptop.
# Above ~100M params the per-step time blows past a minute on consumer CPUs
# and people lose patience long before the loss curve goes anywhere useful.
CPU_MAX_PARAMS = 100_000_000


@dataclass(frozen=True)
class Capability:
    qlora_max_params: int
    lora_max_params: int
    full_ft_max_params: int
    notes: str
    warning_codes: tuple[str, ...] = ()
    cpu_max_params: int = 0


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
            warning_codes=("shared_memory_slow",),
        )

    effective_vram = vram_gb * 0.75 if memory_kind == "unified" else vram_gb

    if effective_vram < _TIERS[0][0]:
        return Capability(
            qlora_max_params=1_000_000_000,
            lora_max_params=350_000_000,
            full_ft_max_params=50_000_000,
            notes="Below 8 GB — only very small models / tiny LoRAs.",
            warning_codes=("below_min_vram",),
        )

    chosen = _TIERS[0]
    for tier in _TIERS:
        if effective_vram >= tier[0]:
            chosen = tier

    note = (
        f"Apple unified memory — using 75% of {vram_gb:.0f} GB ({effective_vram:.0f} GB effective)."
        if memory_kind == "unified" else ""
    )
    return Capability(
        qlora_max_params=chosen[1],
        lora_max_params=chosen[2],
        full_ft_max_params=chosen[3],
        notes=note,
        warning_codes=("unified_memory_overhead",) if memory_kind == "unified" else (),
    )


def capabilities_for_cpu() -> Capability:
    """Capability for the CPU pseudo-device.

    GPU-tier numbers are zero — picking CPU + LoRA + a 7B model in the UI just
    isn't going to work. ``cpu_max_params`` is what the picker reads to decide
    which entries are eligible.
    """
    return Capability(
        qlora_max_params=0,
        lora_max_params=0,
        full_ft_max_params=0,
        notes=(
            "CPU training is hard-capped at ~100M params. "
            "Expect minutes per step on a stock laptop — useful for tiny "
            "models, smoke tests, and demos."
        ),
        warning_codes=("cpu_only_slow",),
        cpu_max_params=CPU_MAX_PARAMS,
    )
