# Supported hardware (v1.0)

LLM-Chain v1.0 supports two backends: NVIDIA CUDA and Apple Silicon MLX. The sidecar probes your machine at startup and the UI grays out models that won't fit. AMD ROCm, Intel XPU, and CPU-only training are deferred to v1.1.

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

## Coming in v1.1

AMD ROCm (RX 7000/9000 on Linux + Windows), Intel XPU (Arc + IPEX-LLM), CPU fallback for ≤100M models.
