# LLM-Chain

Open-source desktop app to train your own LLM locally on your own hardware. Pick a base model, pick a dataset, run a LoRA / QLoRA fine-tune — the UI gates choices to what your machine can actually handle.

**Status:** alpha. Latest tagged release: `v0.1.0-alpha.5.2`. The `main` branch carries the post-alpha.5 feature batch (dataset workshop, synthetic data, A/B comparator, scheduled runs, run notes, curated dataset library, recipes, multi-adapter chat, Ollama integration, DPO, distillation, adapter merge, cloud burst scaffolding, leaderboard submit, folder watch, notifications, multi-window, adapter diff, portable bundles) — see [`docs/development.md`](docs/development.md) for the full list. Next release `v0.1.0-alpha.6` is in CI.

## Supported hardware

| Platform | Status |
| --- | --- |
| Apple Silicon (macOS, MLX) | shipped — DMG bundle |
| NVIDIA CUDA (Linux + Windows) | shipped — MSI / NSIS bundle on Windows; from-source on Linux today (Linux `.deb` / `.rpm` lands with `v0.1.0-alpha.6`) |
| AMD ROCm (Linux + WSL2) | experimental opt-in shipped; native Linux quickstart at [`docs/amd-rocm-quickstart.md`](docs/amd-rocm-quickstart.md), Windows WSL2 walkthrough at [`docs/amd-rocm-wsl2-setup.md`](docs/amd-rocm-wsl2-setup.md). Hardware validation pending. |
| CPU (≤100M models) | shipped |
| Intel XPU (IPEX-LLM) | parked — needs hardware |

See [`docs/supported-hardware.md`](docs/supported-hardware.md) for VRAM/RAM tier expectations.

## What's shipped

The foundation (v1.0) plus four feature batches (Tier A / B / C / D). Highlights:

**Training & evaluation**
- Hardware probe + capability gating (matches model size to your VRAM / unified memory).
- LoRA / QLoRA on NVIDIA CUDA + Apple Silicon MLX; LoRA on AMD ROCm (experimental, opt-in).
- DPO preference fine-tuning via TRL (CUDA / CPU / ROCm).
- Inline knowledge distillation (teacher + student loss).
- Eval suite with per-family default prompts (Qwen3 / SmolLM / Phi / Mistral / Pythia / OLMo / Llama / Gemma / DeepSeek / Qwen2-VL / TinyLlama).
- A/B prompt comparator with quantitative scoring.
- Compare two runs with overlaid loss curves; LR finder with a recommended-LR banner.
- Resume training from a checkpoint; merge multiple LoRA adapters into a new run.

**Datasets**
- Loaders: JSONL chat, CSV, folder of `.txt` files, Hugging Face Hub.
- Dataset workshop: paste rows → schema detection → cleaning → save as JSONL.
- Synthetic data generator: point a chat-tuned model at a topic, generate `(prompt, response)` pairs.
- Curated HF dataset library with one-click download into JSONL.

**UX**
- Live training events (loss / lr / step) over Server-Sent Events.
- HF download progress streamed through the same channel.
- Inference playground with single-slot LRU model cache and SSE streaming.
- Multi-adapter chat: load base once, swap adapters per turn.
- One-click recipes (e.g. "train a customer-support assistant").
- Run notes (markdown attached per run); schedule a run for later.
- Reveal output adapter in Finder/Explorer; cancel mid-step; cancel mid-generation.
- Folder watcher for new datasets; system notifications on long-run completion; multi-window.
- Adapter library with bulk delete; auto-cleanup of old runs by age + status.
- Adapter diff between two runs.

**Export & sharing**
- GGUF export (peft merge + llama.cpp convert).
- Hugging Face Hub push.
- One-click Ollama registration.
- Portable `.llmchain` bundles.
- Opt-in submit-to-leaderboard (phase 1 — no public board UI yet).

For an overview of how a training run flows through the system, see [`docs/data-flow.md`](docs/data-flow.md).

## Still parked / experimental

- AMD ROCm trainer: code path is fully wired; awaiting hardware validation. Run from source today via [`docs/amd-rocm-quickstart.md`](docs/amd-rocm-quickstart.md).
- Cloud burst (provider SDKs deferred): scaffolding shipped, Modal / RunPod / Lambda SDK wiring lands per-provider.
- VLM playground + eval (image input UI).
- Pretraining from scratch.
- Intel XPU.

## Quick start

Download the latest installer from [Releases](https://github.com/Drake0306/LLM-Chain/releases/latest):
- **macOS (Apple Silicon)** → `LLM-Chain_<version>_aarch64.dmg` → drag to Applications.
- **Windows (x64)** → `LLM-Chain_<version>_x64_en-US.msi`.
- **Linux (x86_64)** → `.deb` (Ubuntu / Mint / Debian / Pop) or `.rpm` (Fedora / RHEL / openSUSE) — first published with `v0.1.0-alpha.6`. Until then, run from source — see [`setup.md`](setup.md). AMD GPU users go through [`docs/amd-rocm-quickstart.md`](docs/amd-rocm-quickstart.md).

The macOS / Windows binaries are unsigned in the alpha; expect a one-time Gatekeeper / SmartScreen warning on first launch. See [`setup.md`](setup.md) for the full install + Dock-pin walkthrough.

For dev setup see [`docs/development.md`](docs/development.md). A 10-row demo dataset lives at [`examples/tiny-chat.jsonl`](examples/tiny-chat.jsonl) so you can try the Train flow on first launch.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Bugs go through the [issue templates](.github/ISSUE_TEMPLATE/).

## License

Apache 2.0. See [`LICENSE`](LICENSE).
