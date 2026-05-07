# LLM-Chain

Open-source desktop app to train your own LLM locally on your own hardware. Pick a base model, pick a dataset, run a LoRA / QLoRA fine-tune — the UI gates choices to what your machine can actually handle.

**Status:** v1.0 in development.

## Supported hardware

| Platform | v1.0 | v1.1 (planned) | v1.2 (planned) |
| --- | --- | --- | --- |
| NVIDIA CUDA (Linux + Windows) | yes | — | — |
| Apple Silicon (macOS, MLX) | yes | — | — |
| AMD ROCm | — | yes | — |
| Intel XPU (IPEX-LLM) | — | yes | — |
| CPU (≤100M models) | — | yes | — |

See [`docs/supported-hardware.md`](docs/supported-hardware.md) for VRAM/RAM tier expectations.

## What v1.0 ships

- Hardware probe + capability gating (matches model size to your VRAM/unified memory).
- Curated Apache/MIT model allowlist (Pythia, SmolLM2, Qwen3, Mistral, Phi-4-mini, OLMo).
- Dataset loaders: JSONL chat, CSV, folder of .txt files, Hugging Face Hub.
- LoRA / QLoRA fine-tune via Hugging Face transformers + peft (CUDA) or `mlx_lm.lora` (Apple Silicon).
- Live training events (loss / lr / step) over Server-Sent Events, rendered as a chart + log tail.
- Local-only checkpoint output (no cloud / Hub push yet).

## What v1.1 will add (separate plan)

AMD ROCm + Intel XPU + CPU fallback, Unsloth on NVIDIA, restricted-license toggle (Llama / Gemma / DeepSeek base) with inline warnings, GGUF export, optional Hugging Face Hub push, opt-in anonymous telemetry.

## What v1.2 will add (separate plan)

Pretraining-from-scratch (nanoGPT/nanochat), multimodal training (Qwen2.5-VL, Idefics2/3, Phi-4-multimodal), image-folder + image+caption parquet dataset loaders.

## Quick start

Once releases ship, download the `.dmg` (macOS) or `.msi` (Windows) from the Releases page and launch it.

For now, see [`setup.md`](setup.md) (sidecar smoke test) and [`docs/development.md`](docs/development.md) (full dev setup including the Tauri shell). A 10-row demo dataset lives at [`examples/tiny-chat.jsonl`](examples/tiny-chat.jsonl) so you can try the Train flow on first launch.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Bugs go through the [issue templates](.github/ISSUE_TEMPLATE/).

## License

Apache 2.0. See [`LICENSE`](LICENSE).
