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
- Hugging Face download progress streamed through the same channel — see weights pull in real time before training starts.
- Cancel a running fine-tune from the Run screen; the sidecar honors the signal and marks the run `canceled`.
- Reveal an output adapter directly in Finder/Explorer from the Run detail view.
- Settings screen for default backend (auto / cuda / mlx), default dataset format, and an override for the runs root directory (`LLM_CHAIN_RUNS_DIR`).
- Resilient SSE: connection drops show a "reconnecting…" indicator and the executor refuses to re-run terminal or in-flight runs on auto-reconnect.
- Local-only checkpoint output (no cloud / Hub push yet).

For an overview of how a training run flows through the system, see [`docs/data-flow.md`](docs/data-flow.md).

## What v1.1 will add (separate plan)

AMD ROCm + Intel XPU + CPU fallback, Unsloth on NVIDIA, restricted-license toggle (Llama / Gemma / DeepSeek base) with inline warnings, GGUF export, optional Hugging Face Hub push, opt-in anonymous telemetry.

## What v1.2 will add (separate plan)

Pretraining-from-scratch (nanoGPT/nanochat), multimodal training (Qwen2.5-VL, Idefics2/3, Phi-4-multimodal), image-folder + image+caption parquet dataset loaders.

## Quick start

Download the latest installer from [Releases](https://github.com/Drake0306/LLM-Chain/releases/latest):
- **macOS (Apple Silicon)** → `LLM-Chain_<version>_aarch64.dmg` → drag to Applications.
- **Windows (x64)** → `LLM-Chain_<version>_x64_en-US.msi`.

Both binaries are unsigned in the alpha; expect a one-time Gatekeeper / SmartScreen warning on first launch. See [`setup.md`](setup.md) for the full install + Dock-pin walkthrough.

For dev setup see [`docs/development.md`](docs/development.md). A 10-row demo dataset lives at [`examples/tiny-chat.jsonl`](examples/tiny-chat.jsonl) so you can try the Train flow on first launch.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Bugs go through the [issue templates](.github/ISSUE_TEMPLATE/).

## License

Apache 2.0. See [`LICENSE`](LICENSE).
