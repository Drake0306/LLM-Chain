# LLM-Chain — local setup

Five-minute tour. Full developer documentation lives in [`docs/development.md`](docs/development.md); supported hardware in [`docs/supported-hardware.md`](docs/supported-hardware.md).

## Prereqs

- Python 3.11.x (the sidecar pins `>=3.11,<3.12`)
- Node 20+ and npm
- Rust stable (`rustup default stable`)
- macOS arm64, Linux x86_64, or Windows x86_64

## Install

From the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate                    # .venv\Scripts\activate on Windows
pip install -e './sidecar[dev]'

# Optional platform extras (training won't actually run without one of these):
pip install -e './sidecar[dev,mlx]'          # macOS — Apple Silicon
pip install -e './sidecar[dev,cuda]'         # NVIDIA

cd apps/desktop && npm install && cd ../..
```

Quote `'./sidecar[dev]'` — zsh treats unquoted brackets as a glob.

## Smoke-test the sidecar

```bash
cd sidecar && pytest -v        # 38 fast tests; slow real-training tests gated behind -m slow
```

Boot the FastAPI server alone:

```bash
uvicorn llm_chain_sidecar.main:app --host 127.0.0.1 --port 8000
# in another shell:
curl localhost:8000/api/hardware
curl 'localhost:8000/api/models?max_params=2000000000'
```

## Run the full app

```bash
./scripts/build-sidecar.sh --dev             # one-time: thin wrapper around .venv
cd apps/desktop && npm run tauri dev          # opens the desktop window
```

Click through Dashboard → Models → Dataset → Train → Runs. The training run streams loss back to the loss chart in real time over SSE.

## Probe your hardware from Python (no UI)

```bash
python -c "
from llm_chain_sidecar.hardware import probe_hardware
from llm_chain_sidecar.hardware.capabilities import capabilities_for_vram
report = probe_hardware()
for d in report.devices:
    cap = capabilities_for_vram(d.vram_gb, d.memory_kind)
    print(f'{d.name} ({d.vram_gb} GB, {d.memory_kind}) → QLoRA up to {cap.qlora_max_params/1e9:.1f}B')
"
```

## Build releases

CI builds Mac DMG + Windows MSI on `v*` tag pushes (`.github/workflows/release.yml`). To build locally:

```bash
./scripts/build-sidecar.sh                    # real PyInstaller binary (~10 min)
cd apps/desktop && npm run tauri build
```

## Where things live

```
apps/desktop/         Tauri 2 shell + React UI
sidecar/              Python sidecar (FastAPI + ML)
scripts/              Sidecar build scripts
docs/                 Dev + hardware docs + plans
```
