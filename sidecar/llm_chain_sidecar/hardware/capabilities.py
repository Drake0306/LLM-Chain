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


def capabilities_for_vram(
    vram_gb: float,
    memory_kind: MemoryKind = "dedicated",
    available_vram_gb: float | None = None,
) -> Capability:
    """Return the max trainable params at each technique for a given VRAM tier.

    Apple Silicon "unified" memory really IS GPU-accessible at full speed — count it.
    PC "shared GPU memory" is system DDR over PCIe at ~30-60 GB/s, ~20x slower than
    real VRAM. The NVIDIA Windows driver silently spills there and tanks throughput
    10-50x rather than OOMing. Treat shared memory as effectively zero for training.

    ``available_vram_gb`` is only meaningful for ``unified`` memory: a Mac
    with 16 GB total but 8 GB already in use by other apps shouldn't
    advertise the 12 GB-effective tier (16 × 0.75) because the model load
    will OOM. We cap the effective number by what's actually free.
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

    if memory_kind == "unified":
        theoretical = vram_gb * 0.75
        # If the probe gave us a live "available" reading, cap by it.
        # available_vram_gb already reflects total - used, so no further
        # multiplication. The min() picks whichever is more restrictive.
        if available_vram_gb is not None:
            effective_vram = min(theoretical, available_vram_gb)
        else:
            effective_vram = theoretical
    else:
        effective_vram = vram_gb

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

    if memory_kind == "unified":
        if available_vram_gb is not None and available_vram_gb < vram_gb * 0.75:
            note = (
                f"Apple unified memory — {vram_gb:.0f} GB total, "
                f"{available_vram_gb:.0f} GB available right now "
                f"(other apps are using the rest). Using "
                f"{effective_vram:.0f} GB effective."
            )
        else:
            note = (
                f"Apple unified memory — using 75% of {vram_gb:.0f} GB "
                f"({effective_vram:.0f} GB effective)."
            )
    else:
        note = ""
    return Capability(
        qlora_max_params=chosen[1],
        lora_max_params=chosen[2],
        full_ft_max_params=chosen[3],
        notes=note,
        warning_codes=("unified_memory_overhead",) if memory_kind == "unified" else (),
    )


def capabilities_for_amd_vram(vram_gb: float) -> Capability:
    """AMD ROCm capability gate.

    Mirrors the dedicated-VRAM tier table — same memory math applies in
    principle — but always carries the ``rocm_unverified`` warning. We have
    not yet validated a real LoRA/QLoRA run on ROCm hardware, so the UI
    surfaces this as experimental and the HfRocmTrainer raises on instantiation
    with a request to report results.
    """
    base = capabilities_for_vram(vram_gb, memory_kind="dedicated")
    note = (
        "AMD ROCm support is experimental — capacity numbers are inherited "
        "from the NVIDIA tier table and have NOT been validated on AMD "
        "hardware. Please report what works (and what doesn't) at "
        "https://github.com/Drake0306/LLM-Chain/issues."
    )
    if base.notes:
        note = f"{base.notes} {note}"
    return Capability(
        qlora_max_params=base.qlora_max_params,
        lora_max_params=base.lora_max_params,
        full_ft_max_params=base.full_ft_max_params,
        notes=note,
        warning_codes=("rocm_unverified", *base.warning_codes),
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
