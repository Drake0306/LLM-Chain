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

## AMD ROCm (experimental opt-in)

The probe detects AMD GPUs via `torch.version.hip` and the Dashboard renders them with an amber chip. The capability gate reuses the dedicated VRAM tier table above (a 24 GB Radeon advertises the same QLoRA cap as a 24 GB NVIDIA card) but always carries a `rocm_unverified` warning code. `HfRocmTrainer` is a real subclass of `HfCudaTrainer` (HIP reuses CUDA's API surface), gated by an env-var opt-in to keep AMD-curious users from running unvalidated training without realising it.

**Opting in on real AMD hardware:** set `LLM_CHAIN_ROCM_EXPERIMENTAL=1` before launching the sidecar. The card becomes selectable, the chip flips to **"experimental ARMED — LoRA only, please report results"**, and LoRA runs go through under a loud warning in the sidecar log. QLoRA still refuses because `bitsandbytes` is CUDA-only.

**Distro-specific quickstarts:**

- **Native Linux** (Fedora / Ubuntu / Mint / Arch / openSUSE / RHEL): [`amd-rocm-quickstart.md`](amd-rocm-quickstart.md) — installs ROCm 6.3+, the ROCm-flavored PyTorch wheel, and walks you through milestones 1-4 (probe sees the GPU → UI shows it → LoRA smoke test passes → end-to-end UI flow).
- **Windows + WSL2**: [`amd-rocm-wsl2-setup.md`](amd-rocm-wsl2-setup.md) — same milestone framework, extra steps for Adrenalin Pro driver + WSL2 + `--usecase=wsl,rocm`.

**To make the probe see your AMD GPU at all** (the prerequisite for either quickstart):

- **Linux:** install ROCm 6.3+ for your distro, then `pip install --pre torch --index-url https://download.pytorch.org/whl/rocm6.3`. Skip the `[cuda]` extra; `bitsandbytes` is CUDA-only.
- **Windows:** there is no native PyTorch+ROCm wheel for Windows. Use WSL2 (Ubuntu 22.04 / 24.04) and install the Linux ROCm wheel inside WSL — see the WSL2 walkthrough above. The Microsoft DirectML stack (`torch-directml`) is a separate code path and is **not** detected by the probe.
- **macOS:** ROCm is Linux-only — there is nothing to install.

**ROCm-version-by-card cheat sheet:** RDNA 4 (RX 9070 / 9070 XT) needs ROCm 6.3+; RDNA 3 (RX 7900 family) wants 6.0+; RDNA 2 (RX 6000 family) is officially unsupported and only kind-of works via `HSA_OVERRIDE_GFX_VERSION`; CDNA / Instinct accelerators run on 5.x.

If you have AMD silicon, please open an issue at <https://github.com/Drake0306/LLM-Chain/issues> with the `/api/hardware` JSON and any smoke-test results — that's the gating signal before we promote `HfRocmTrainer` out of "experimental" and ship a dedicated ROCm `.deb` / `.rpm`.

## Still parked

Intel XPU (Arc + IPEX-LLM) — needs hardware we don't have access to.
