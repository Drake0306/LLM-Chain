# Supported hardware

LLM-Chain ships fully-validated training on two backends: **NVIDIA CUDA** and **Apple Silicon MLX**. The sidecar probes your machine at startup and the UI grays out models that won't fit. **CPU fallback** (≤100M models) shipped in v1.1. **AMD ROCm** is detected and surfaced in the UI as experimental — see [AMD ROCm (experimental)](#amd-rocm-experimental) below. Intel XPU is still parked.

## VRAM-tier capability gates

The numbers below are the maximum trainable parameter count for each technique at a given VRAM (NVIDIA) or unified memory (Apple Silicon) size. Implementation: [`sidecar/llm_chain_sidecar/hardware/capabilities.py`](../sidecar/llm_chain_sidecar/hardware/capabilities.py).

| Effective VRAM | QLoRA | LoRA | Full FT |
| ---: | ---: | ---: | ---: |
| 8 GB | 7B | 3B | 125M |
| 12 GB | 13B | 7B | 350M |
| 16 GB | 13B | 13B | 500M |
| 24 GB | 34B | 13B | 1.3B |
| 32 GB | 34B | 13B | 1.5B |
| 48 GB | 70B | 30B | 3B |
| 64 GB | 70B | 30B | 3B |
| 128 GB | 70B | 70B | 7B |

Below 8 GB the gate falls back to a deliberately small budget (1B QLoRA / 350M LoRA / 50M full FT) and a `below_min_vram` warning code.

## Memory kinds

Each detected GPU device carries a `memory_kind`:

- **`dedicated`** — NVIDIA discrete VRAM. Use the table above as-is.
- **`unified`** — Apple Silicon shared system memory. The gate multiplies the reported pool by **0.75** before looking up the tier (the OS holds the rest for itself and overflow paging is slow). A 64 GB Mac is gated as a 48 GB tier device.
- **`shared`** — PC "shared GPU memory" backed by system DDR over PCIe. ~20× slower than real VRAM and the NVIDIA Windows driver silently spills there, tanking throughput. The gate ignores this path entirely and returns the below-8-GB minimal budget plus a `shared_memory_slow` warning.

## Devices we test against

- Apple Silicon: M1 Pro 16 GB, M2 Max 32 GB, M3 Max 64 GB, M2 Ultra 192 GB (manual smoke).
- NVIDIA: RTX 4090 24 GB, RTX 4080 16 GB, RTX 3090 24 GB, A100 40/80 GB (manual smoke).

The slow integration test (`pytest -m slow`) actually fine-tunes a tiny model and asserts events stream through. Run it on an actual GPU host before shipping a release candidate.

## What's not gated yet

- **Multi-GPU.** v1.0 picks the first detected device. Multi-GPU LoRA via `accelerate` lands in v1.1.
- **Quantized weights.** The tiers assume bf16 base + LoRA adapters. If you load 4-bit base via `bitsandbytes` (QLoRA path) the budget is roughly correct; for 8-bit it's conservative.
- **Activation memory.** Long sequences (>8K context) eat the full-FT budget faster than the table suggests. The gate doesn't know your `max_seq_len` yet.

## AMD ROCm (experimental)

The probe detects AMD GPUs via `torch.version.hip` and the Dashboard renders them with an amber **"experimental — not yet validated on hardware"** chip. The capability gate reuses the dedicated VRAM tier table above (a 24 GB Radeon advertises the same QLoRA cap as a 24 GB NVIDIA card) but always carries a `rocm_unverified` warning code, and `HfRocmTrainer` raises `NotImplementedError` on instantiation rather than silently kicking off a run we can't vouch for.

**To make the probe see your AMD GPU:**

- **Linux:** install a ROCm build of PyTorch — `pip install --pre torch --index-url https://download.pytorch.org/whl/rocm6.2`. Skip the `[cuda]` extra; `bitsandbytes` is CUDA-only.
- **Windows:** there is no native PyTorch+ROCm wheel for Windows. Use **WSL2 (Ubuntu 22.04 / 24.04)** and install the Linux ROCm wheel inside WSL — see the full step-by-step at [`amd-rocm-wsl2-setup.md`](amd-rocm-wsl2-setup.md). The Microsoft DirectML stack (`torch-directml`) is a separate code path and is **not** detected by the probe.
- **macOS:** ROCm is Linux-only — there is nothing to install.

If you have AMD silicon, please open an issue at <https://github.com/Drake0306/LLM-Chain/issues> with what worked and what didn't — that's the gating step before we promote `HfRocmTrainer` past the stub.

## Still parked

Intel XPU (Arc + IPEX-LLM) — needs hardware we don't have access to.
