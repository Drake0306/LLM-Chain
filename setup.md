# LLM-Chain — local setup

Repo: https://github.com/Drake0306/LLM-Chain

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

## GitHub Actions workflows

Two workflows live under `.github/workflows/`. Watch them at
https://github.com/Drake0306/LLM-Chain/actions.

### `ci.yml` — every push to `main`, every PR

| Matrix OS | What runs |
| --- | --- |
| `ubuntu-latest` | `pip install -e ./sidecar[dev]` then `pytest sidecar/tests -v` |
| `macos-14` | same |
| `windows-latest` | same |

Default `pytest` excludes the slow real-training tests (`-m 'not slow'`),
so this run is fast and free.

### `release.yml` — `v*` tag pushes (or manual `workflow_dispatch`)

| Matrix OS | Target | Sidecar extra | Bundle |
| --- | --- | --- | --- |
| `macos-14` | `aarch64-apple-darwin` | `mlx` | `.dmg` |
| `windows-latest` | `x86_64-pc-windows-msvc` | `cuda` | `.msi` (+ `.nsis`) |

Each leg installs the sidecar with its platform extras, runs `pytest`,
runs `scripts/build-sidecar.sh` (or `.ps1`) to produce a PyInstaller
binary, then runs `npm run tauri build`. Artifacts upload as
`llm-chain-<target>` on the run page.

### Triggering a release

```bash
git tag -a v0.1.1-alpha -m "release notes here"
git push origin v0.1.1-alpha
```

The first release `v0.1.0-alpha` is already pushed:
https://github.com/Drake0306/LLM-Chain/actions

To trigger a release build without a new tag:

```bash
gh workflow run release.yml --ref main
```

### Running CI locally before pushing

```bash
# what ci.yml runs:
cd sidecar && pytest -v

# what release.yml runs (slow — produces real installer):
./scripts/build-sidecar.sh
cd apps/desktop && npm run tauri build
# outputs land under apps/desktop/src-tauri/target/<triple>/release/bundle/
```

## Where things live

```
apps/desktop/         Tauri 2 shell + React UI
sidecar/              Python sidecar (FastAPI + ML)
scripts/              Sidecar build scripts
docs/                 Dev + hardware docs + plans
.github/workflows/    CI (ci.yml) + release (release.yml)
```
