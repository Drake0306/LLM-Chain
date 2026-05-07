# Local LLM Trainer — Research & Design (Working Draft)

**Date:** 2026-05-07
**Status:** Research complete; user approved all recommendations on 2026-05-07. Decisions locked in Section 7. Proceeding to implementation plan.
**Goal:** Open-source desktop app (Mac + Windows, native window) that lets a user pick a base model, pick a hardware backend, supply a dataset, and train/fine-tune a local LLM — with the UI gating choices to what their hardware can actually do.

This document captures research findings only. No code has been written. The final section lists the decisions that need user input before we write an implementation plan.

---

## 1. The "LangChain" misconception (important)

LangChain — and LlamaIndex, Haystack — are **inference-time** orchestration frameworks (chains, agents, RAG, tool use). They have **no trainer, no optimizer loop, no LoRA/PEFT, no training kernels**. The LangChain maintainers themselves (GitHub discussion #21558) tell users to use Hugging Face for fine-tuning.

The actual local fine-tuning stack in 2026:

| Layer | What people actually use |
|---|---|
| Core engine (NVIDIA / AMD / Intel / CPU) | **Hugging Face `transformers` + `trl` + `peft` + `accelerate`** |
| Speed-optimized NVIDIA wrapper | **Unsloth** (Triton kernels, ~70% less VRAM, fastest single-GPU) |
| Config-driven YAML wrapper | **Axolotl** (production-ready, multimodal, RLHF) |
| GUI wrapper | **LLaMA-Factory** (uses Unsloth as backend) |
| PyTorch-native | **torchtune** (FSDP2, DoRA) |
| Apple Silicon native | **MLX-LM** (mandatory on Macs without CUDA) |

**Where LangChain still earns its place in this app:** the *post-training* side. After the user trains a model, LangChain is a clean way to wire up:
- An eval harness (run the new model through eval datasets)
- A "talk to your model" chat playground
- A RAG-over-the-training-data inspector for comparing base vs. fine-tuned outputs
- Dataset-prep helpers (LangSmith trace export, Tuna for synthetic data)

**Recommended framing in the UI:** "Training engine: Hugging Face / Unsloth / MLX-LM. Evaluation & playground: LangChain."

---

## 2. Cross-platform desktop framework

Comparison of realistic options for a Python-ML-backed desktop app on Mac + Windows. Bundle sizes assume PyTorch + CUDA wheels are bundled (which dominates at ~2 GB regardless of shell choice).

| Framework | Shell size | UI tech | IPC to Python ML | Signing/notarize | License | Single-installer pain |
|---|---|---|---|---|---|---|
| **Tauri 2.x** | 3–10 MB | Web (React/Svelte) | Sidecar Python via `externalBin`, talk over stdio or local FastAPI | First-class v2 helpers (`tauri signer`, codesign, signtool) | MIT/Apache | Medium |
| Electron | 120–180 MB | Web | `child_process.spawn` + JSON-RPC or local FastAPI | Mature (`electron-builder`, Squirrel auto-update) | MIT | Low–medium |
| **PySide6 (Qt6)** | 150–300 MB | Native Qt widgets | **In-process** (no IPC) — train in `QThread` | Standard `codesign`/`signtool` via PyInstaller/Briefcase | **LGPL-3** (OSS-friendly, dynamic link) | Low |
| Flet | 40 MB + Python | Flutter widgets | In-process Python | PyInstaller-style | Apache-2.0 | Low |
| pywebview / NiceGUI | 80–250 MB | HTML/JS in OS WebView | In-process Python | PyInstaller-style | BSD-3 / MIT | Low |
| Streamlit / Gradio | n/a | Web | n/a | Not desktop-grade | Apache-2.0 | Wrong tool |
| ~~PyQt6~~ | ~~150–300 MB~~ | ~~Native Qt~~ | ~~In-process~~ | ~~Standard~~ | **GPL-3 or commercial** | **Avoid for permissive OSS** |

### Recommendation

**Top: Tauri 2.x + Python sidecar (FastAPI/uvicorn over stdio or localhost).**
Smallest shell, MIT/Apache, strong v2 signing/auto-update story, web frontend gives us all the dashboards/charts/log streams cleanly (React + Recharts/visx). The 2+ GB PyTorch payload dominates anyway, so Electron's Chromium tax is wasted bytes.

**Runner-up: PySide6 (LGPL).**
If we want a Python-only codebase with no JS/Rust tax, in-process is dramatically simpler — training callbacks update Qt widgets directly, no IPC layer to debug. Trade-off: less modern-looking UI, more boilerplate for charts.

---

## 3. Multi-vendor GPU training support (state of the world, May 2026)

| Backend | macOS | Windows | Linux | 7B LoRA realistic? | Notes |
|---|---|---|---|---|---|
| **Apple MLX (Metal)** | ✅ | – | – | ✅ on 16 GB+ | `mlx-lm` LoRA/QLoRA/full FT all production-ready; M5 GPU neural accelerators land via MLX |
| **NVIDIA CUDA** | – | ✅ | ✅ | ✅ on 12 GB+ | PyTorch + bitsandbytes + Unsloth all native on Windows since 2024–25; CUDA 13 needs Turing+ |
| **AMD ROCm** | – | ✅ (RDNA3/4) | ✅ | ✅ Linux; BF16-only on Win | **Big 2026 change**: ROCm 6.4.4 / 7.x ships official native PyTorch wheels for Windows on RX 7000/9000 + Ryzen AI 300/Max APUs. WSL no longer required. bitsandbytes ROCm port is Linux-only — Windows users get BF16 LoRA but no 4-bit QLoRA (yet) |
| **Intel XPU** | – | partial | ✅ | yes-ish (transition) | IPEX is being retired (March 2026); upstream PyTorch XPU is the future. IPEX-LLM still active; supports LoRA/QLoRA/DPO on Arc A770, B580, integrated Xe |
| **CPU fallback** | ✅ | ✅ | ✅ | ❌ — only <100 M models | Reasonable for pedagogical demos and tiny pretraining only |

**Apple Silicon "unified memory" vs PC "shared GPU memory" — these are NOT the same:**
- **Apple unified memory:** CPU and GPU share one on-package pool at 400–800 GB/s. The GPU genuinely uses it at full speed. On a 128 GB Mac, ~96 GB is GPU-addressable. MLX exploits this; 70B QLoRA is real on 64 GB+.
- **PC "shared GPU memory"** (Windows Task Manager): system DDR5 over PCIe at 30–60 GB/s. ~20× slower than VRAM. NVIDIA Windows driver silently spills here when VRAM fills, causing 10–50× slowdowns instead of OOMs — users think their GPU is broken. **Treat shared GPU memory as zero for gating purposes.** Default to a 0.95 VRAM allocator cap to prevent the spillover trap.

---

## 4. VRAM-to-trainable-params heuristics (the gating logic)

| VRAM | QLoRA (4-bit) | LoRA (BF16) | Full Fine-Tune |
|---|---|---|---|
| 8 GB (RTX 4060) | 7B tight; 3B comfortable | 1–3B only | <125 M |
| 12 GB (RTX 4070, RX 7700 XT) | 7B comfortable; 13B tight | 3B comfortable; 7B tight | <350 M |
| 16 GB (RTX 4080, RX 7800 XT, M-base) | 13B comfortable; 8B easy | 7B comfortable; 13B tight | ~500 M |
| 24 GB (3090/4090, RX 7900 XTX) | 13B comfortable; 30–34B tight | 13B comfortable | 1–1.3B |
| 32 GB unified (M-Pro) | 13B comfortable; 30B feasible | 13B comfortable | ~1.5B |
| 48 GB (RTX 6000 Ada / 2×24) | 70B feasible; 30B easy | 30B comfortable | 3B |
| 64–128 GB unified (M-Max/Ultra) | 70B comfortable | 70B feasible @128 GB | 7B feasible @128 GB |

**Pretraining-from-scratch (nanoGPT-style):**
- 10 M params: trivial on any 8 GB+; minutes to hours
- 50 M: comfortable on 12 GB+; few hours on a 4090
- 125 M (GPT-2 small): 2–4 h on RTX 4090; ~8 h on M-series via MPS
- 350 M – 1B: technically possible on 24 GB but days–weeks; not practical for end users

**Recommendation:** gate pretraining at ≤125 M for 12 GB+, ≤350 M for 24 GB+, and warn that anything above 125 M is overnight-plus.

---

## 5. Model picker — commercially-clean shortlist

License audit. Apache-2.0 / MIT only unless flagged. All hosted on Hugging Face Hub.

### Small (10 M – 500 M) — fine-tune or train from scratch on consumer GPUs
- **Pythia 14M / 70M / 160M / 410M** (EleutherAI) — Apache 2.0, fully open data. Best from-scratch reference tier.
- **SmolLM2 / SmolLM3 small variants** (HF) — Apache 2.0, fully open recipe.
- **Qwen3 0.6B** (Alibaba) — Apache 2.0, no MAU cap, no AUP.
- **Phi-4-mini** (Microsoft) — MIT.
- ⚠️ **Gemma 3 270M/1B** — custom Gemma ToU with prohibited-use policy + downstream pass-through. Avoid unless Gemma 4 lands as Apache.

### Mid (1B – 8B) — primary fine-tuning sweet spot
- **Mistral 7B / Ministral 3B / 8B / 14B** — Apache 2.0
- **Qwen3 1.7B / 4B / 8B** — Apache 2.0
- **Phi-4 / Phi-4-mini-instruct / Phi-4-multimodal** — MIT
- **TinyLlama 1.1B** — Apache 2.0 (stale 2024 but useful baseline)
- ⚠️ **Falcon 3 1B/3B/7B/10B** — TII Falcon-LLM License 2.0 (commercial OK, but custom not OSI-approved)
- ⚠️ **OLMo 2 7B** — Apache 2.0 weights but card says "research/educational"
- ❌ **Llama 3.x 8B / Llama 4** — Llama Community License: >700M MAU clause, "Built with Llama" attribution, training-other-models ban, **EU exclusion for Llama 4 multimodal**. Source-available, not OSI open source. Behind a "restricted license" gate or omit entirely.

### Large (13B – 70B+)
- **Mistral Large 3** (MoE 41B active / 675B total) — Apache 2.0
- **Qwen3 32B / Qwen3.6-27B / Qwen3 235B-A22B** — Apache 2.0
- **OLMo 2 13B / 32B-Instruct** — Apache 2.0
- ⚠️ **DeepSeek V3** — base weights under custom "DeepSeek Model License" (use-based restrictions); the V3-0324 checkpoint moved fully to MIT
- ⚠️ **Falcon 3 (larger)** — TII custom commercial license
- ❌ **Llama 3.1 70B / Llama 4** — same caveats as above

### Multimodal (image + text)
- **Qwen2.5-VL / Qwen3-VL** (3B/7B/32B/72B) — Apache 2.0; cleanest commercial VLM family
- **Idefics2 / Idefics3 8B** (HF) — Apache 2.0
- **Phi-4-multimodal** (Microsoft) — MIT
- ⚠️ **LLaVA-1.5 / LLaVA-NeXT** — code Apache, but weights inherit Llama Community License; prefer LLaVA built on Mistral/Qwen backbones
- ⚠️ **PaliGemma** — Gemma ToU
- **NVIDIA Nemotron Nano Omni** — NVIDIA Open Model License (commercial OK, custom)

### Train-from-scratch frameworks (10 M – 100 M tier)
- **nanoGPT** (Karpathy) — MIT, the standard reference
- **nanochat** (Karpathy, 2025) — MIT, full pretraining → SFT → chat-UI pipeline
- **llm.c** (Karpathy) — MIT, raw C/CUDA, reproduces GPT-2 series
- **GPT-NeoX** (EleutherAI) — Apache 2.0, the framework Pythia was trained with

### Where to host the model list
- **Primary:** Hugging Face Hub. Filter `?license=license:apache-2.0` and `?license=license:mit`.
- **Secondary registries** (mirror HF licensing): Ollama library, LM Studio Model Catalog (good for GGUF), Kaggle Models.
- **No curated permissive-only leaderboard exists.** Open LLM Leaderboard does not filter by license. The app should maintain its own allowlist (the shortlist above) and expose a "show restricted-license models" toggle that adds Llama/Gemma behind clear disclaimers.

---

## 6. Proposed end-to-end architecture (sketch — to be refined after user feedback)

```
┌─────────────────────────────────────────────────────────────┐
│  Tauri 2.x desktop shell  (MIT/Apache, 3–10 MB Rust shell)  │
│  ├─ React + Tailwind + Recharts/visx                        │
│  ├─ Pages: Dashboard / Hardware / Models / Datasets /       │
│  │         Train / Runs / Playground                        │
│  └─ Talks to ↓ over localhost FastAPI (JSON + SSE for logs) │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Python sidecar (PyInstaller / python-build-standalone)     │
│  ├─ FastAPI app                                             │
│  ├─ Hardware probe (torch.cuda / torch.mps / torch.xpu /    │
│  │    rocm-smi / pyobjc Metal query)                        │
│  ├─ Model registry (curated commercial-clean allowlist +    │
│  │    HF Hub search by license)                             │
│  ├─ Dataset loader (HF datasets, JSONL, CSV, image folders) │
│  ├─ Trainer dispatch:                                       │
│  │   ├─ NVIDIA / AMD-Linux → Unsloth → HF trl/peft         │
│  │   ├─ AMD-Win / Intel    → HF trl/peft (BF16 LoRA)       │
│  │   ├─ Apple Silicon       → MLX-LM                        │
│  │   └─ CPU fallback        → HF trl/peft (≤100 M only)    │
│  ├─ Run manager (process supervision, checkpoints, logs)    │
│  └─ Eval/playground (LangChain ChatModel over local model)  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Local filesystem                                           │
│  ~/.llm-chain/{models, datasets, runs, checkpoints}         │
└─────────────────────────────────────────────────────────────┘
```

### Hardware-detection → trainer-selection logic
1. On launch, probe GPUs and report VRAM / unified memory / OS / driver.
2. Map detected backend to trainer:
   - macOS Apple Silicon → MLX-LM
   - Linux + NVIDIA → Unsloth (fastest) with HF trl/peft fallback
   - Windows + NVIDIA → Unsloth (Triton on Windows still slightly behind Linux; offer WSL2 toggle)
   - Linux + AMD RDNA3/4 → HF trl/peft + ROCm bitsandbytes
   - Windows + AMD RDNA3/4 → HF trl/peft (BF16 LoRA only; no 4-bit yet)
   - Intel XPU → IPEX-LLM (note ongoing transition to upstream PyTorch XPU)
   - Anything else → CPU mode, ≤100 M model gate
3. Use the VRAM-tier table to grey-out / disable model+technique combinations that won't fit.

---

## 7. Decisions (locked 2026-05-07)

User approved all recommendations and chose "consider all cases" for the multi-choice questions.

| # | Decision | Rationale |
|---|---|---|
| 1 | **Desktop framework: Tauri 2.x + Python sidecar (FastAPI)** | Smallest shell (3–10 MB Rust), MIT/Apache, first-class signing/auto-update, web frontend handles charts/log streams cleanly. PyTorch payload dominates anyway. |
| 2 | **Restricted-license models: show behind a toggle, with inline warning badge** | Best of both worlds — researchers and commercially-cleared users get access; default users see Apache/MIT only. Each restricted model shows the specific clause (MAU / EU exclusion / AUP / custom). |
| 3 | **Pretraining-from-scratch: included in v1** | The user's brief explicitly mentioned "10 million, 20 million" parameter targets — that's pretraining territory. Wire nanoGPT/nanochat behind the same hardware-gating UI. |
| 4 | **Multimodal (image+text): included in v1** | Brief explicitly called out images, text, and both. Start with Qwen2.5-VL / Idefics2 / Phi-4-multimodal; reuse same trainer dispatch with vision encoder branch. Higher VRAM gates apply. |
| 5 | **Export targets: all three — local, GGUF for Ollama/LM Studio, push-to-HF-Hub** | Local is free; GGUF export is a one-liner via llama.cpp converters; HF Hub push is `huggingface_hub` SDK. |
| 6 | **Dataset formats v1: JSONL (chat), CSV, plain-text directory, HF datasets ID, image folder, image+caption parquet** | Covers all modalities the brief listed. |
| 7 | **Telemetry: default-off, opt-in only** | OSS norm. If user opts in, anonymous training-run metadata only (hardware, model size, success/fail) — never datasets or model weights. |

### Implied scope expansion: phased v1 rollout

Including pretraining + multimodal + all export targets in v1 makes the surface area large. Recommend an internal **three-phase v1** so we can ship something usable along the way without compromising scope:

- **v1.0 — "It trains" (text fine-tuning, single backend per OS):**
  Tauri shell + Python sidecar, hardware probe, model picker (Apache/MIT only), LoRA/QLoRA fine-tune via HF trl/peft (NVIDIA + Apple MLX), JSONL/CSV/text-dir/HF datasets, local-only export.

- **v1.1 — "It trains on everything" (full backend matrix + restricted models):**
  Adds AMD ROCm (Linux + Windows), Intel XPU (IPEX-LLM), Unsloth integration on NVIDIA, restricted-license toggle with warnings, GGUF + HF Hub export.

- **v1.2 — "It trains anything" (pretraining + multimodal):**
  Adds nanoGPT/nanochat from-scratch path, multimodal models (Qwen-VL, Idefics, Phi-4-multimodal), image-folder + image+caption datasets.

Each phase is independently shippable and reviewable.

---

## 8. Non-goals (explicit YAGNI)

- No authentication, no accounts, no cloud anything.
- No multi-node / multi-machine distributed training in v1.
- No RLHF / DPO in v1 (SFT + LoRA / QLoRA only). Add later via TRL.
- No model serving as an API endpoint (the playground talks in-process; if users want serving, point them at Ollama/LM Studio/vLLM).
- No mobile / web targets.
