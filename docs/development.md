# Development setup

## Prerequisites

- **Python 3.11.x** (the sidecar pins `>=3.11,<3.12`).
- **Node 20+** and **npm**.
- **Rust stable** (install via [rustup](https://rustup.rs/)).
- macOS arm64, Linux x86_64, or Windows x86_64.

## One-time setup

```bash
# 1. Clone and enter the repo.
git clone https://github.com/<owner>/LLM-Chain.git
cd LLM-Chain

# 2. Create a Python venv and install the sidecar in editable mode.
python3.11 -m venv .venv
source .venv/bin/activate                              # or .venv\Scripts\activate on Windows
pip install -e './sidecar[dev]'

# 3. (Optional) Install the platform-specific training extras.
#    macOS: pulls mlx + mlx-lm.
pip install -e './sidecar[dev,mlx]'
#    NVIDIA: pulls bitsandbytes for QLoRA.
pip install -e './sidecar[dev,cuda]'

# 4. Install desktop dependencies.
cd apps/desktop
npm install
cd ../..

# 5. Build the sidecar binary the Tauri shell expects.
#    --dev produces a thin wrapper around the venv (fast, non-portable).
#    Without --dev, runs PyInstaller (slow, but produces a real release binary).
./scripts/build-sidecar.sh --dev
```

## Run the app in dev mode

```bash
cd apps/desktop
npm run tauri dev
```

The Tauri window opens, spawns the sidecar, parses the port from its first stdout line, and the React UI hits `http://127.0.0.1:<port>`.

## Tests

```bash
# Python sidecar (fast — slow tests excluded by default)
pytest sidecar -v

# Real training smoke tests (skipped without GPU/MLX)
pytest sidecar -v -m slow

# Frontend
cd apps/desktop && npm test

# Rust shell
cd apps/desktop/src-tauri && cargo check
```

## Building installers locally

```bash
./scripts/build-sidecar.sh                      # macOS / Linux: real PyInstaller binary
pwsh ./scripts/build-sidecar.ps1                # Windows
cd apps/desktop && npm run tauri build
```

Outputs land under `apps/desktop/src-tauri/target/<triple>/release/bundle/`. CI runs the same steps via `.github/workflows/release.yml` on `v*` tag pushes.

## Repo layout

```
apps/desktop/         Tauri 2 shell + React/TS frontend
  src/                React app
    api/              Typed sidecar client + sidecar-port hook
    screens/          Dashboard, ModelPicker, DatasetPicker, Train, Runs, Settings
    state/            Selection context + persisted Settings
  src-tauri/          Rust shell, capabilities, sidecar wiring + desktop-settings.json reader
sidecar/              Python sidecar
  llm_chain_sidecar/
    api/              FastAPI routes (incl. SSE stream + cancel)
    hardware/         Probe + VRAM-tier capability gating
    models/           Apache/MIT allowlist registry
    datasets/         JSONL/CSV/text-dir/HF loaders
    runs/             Run store + executor (cancellation tokens, reconnect guards)
    trainers/         HF/CUDA + MLX backends; hf_progress.py bridges tqdm to SSE
docs/plans/           Implementation plan
docs/data-flow.md     End-to-end flow + Mermaid diagrams of the runtime
scripts/              Sidecar build scripts (PyInstaller + dev wrapper)
.github/workflows/    CI + release
```

For an end-to-end picture (how a click on **Start training** turns into SSE events that drive the chart), read [`data-flow.md`](data-flow.md).

## Troubleshooting

- **`zsh: no matches found: ./sidecar[dev]`** — quote the path: `pip install -e './sidecar[dev]'`. Zsh treats unquoted brackets as a glob.
- **`resource path 'binaries/llm-chain-sidecar-...' doesn't exist`** when `cargo check` runs — Tauri requires the sidecar binary to exist at build time. Run `./scripts/build-sidecar.sh --dev` once.
- **Port collision** — the sidecar picks a free port at startup; the Rust shell parses `LLM_CHAIN_SIDECAR_PORT=<n>` from stdout. If `useApiClient` returns null forever, check the dev-window devtools for invoke errors.
