# LLM-Chain — local setup

Repo: https://github.com/Drake0306/LLM-Chain

Open-source desktop app to fine-tune an LLM on your own machine. Tauri 2 shell + Python sidecar; LoRA / QLoRA on NVIDIA CUDA and Apple Silicon MLX.

For deeper dev docs see [`docs/development.md`](docs/development.md). For the VRAM-tier table see [`docs/supported-hardware.md`](docs/supported-hardware.md).

---

## How to run it

### 0. Prereqs

- **Python 3.11.x** (the sidecar pins `>=3.11,<3.12`)
- **Node 20+** and **npm**
- **Rust stable** (`rustup default stable`)
- macOS arm64, Linux x86_64, or Windows x86_64

Check with:

```bash
python3.11 --version    # should print 3.11.x
node --version          # should print v20+
rustc --version
```

### 1. Clone

```bash
git clone https://github.com/Drake0306/LLM-Chain.git
cd LLM-Chain
```

### 2. Set up the Python sidecar

```bash
python3.11 -m venv .venv
source .venv/bin/activate                   # on Windows: .venv\Scripts\activate
pip install -e './sidecar[dev]'
```

Quote `'./sidecar[dev]'` — zsh treats unquoted brackets as a glob.

To actually train (rather than just run the UI shell), also install the platform extra for your hardware:

```bash
pip install -e './sidecar[dev,mlx]'         # macOS Apple Silicon
pip install -e './sidecar[dev,cuda]'        # NVIDIA Linux/Windows
```

### 3. Verify the sidecar

```bash
cd sidecar && pytest -v && cd ..
```

You should see **40 passed, 2 deselected** (the 2 deselected are the slow real-training tests, gated behind `-m slow`).

### 4. Build the sidecar binary the Tauri shell expects

```bash
./scripts/build-sidecar.sh --dev
```

This writes a thin shell wrapper at `apps/desktop/src-tauri/binaries/llm-chain-sidecar-<your-triple>` that just `exec`s `python -m llm_chain_sidecar.main` from your `.venv`. Fast (~1 second) and only valid on this machine. For a portable binary, omit `--dev` — that runs PyInstaller and takes ~10 minutes.

### 5. Install desktop dependencies

```bash
cd apps/desktop && npm install && cd ..
```

### 6. Launch the app

```bash
cd apps/desktop && npm run tauri dev
```

The Tauri window opens, spawns the sidecar, parses its port off stdout, and the React UI hits `http://127.0.0.1:<port>`. First launch compiles the Rust shell (~1 minute); reloads after that are instant.

### 7. Click through the flow

1. **Dashboard** — your hardware report. Pick a device (CUDA or MLX); CPU is non-selectable.
2. **Model** — picks from the curated Apache/MIT allowlist, gated by your selected device's QLoRA cap. Toggle QLoRA / LoRA at the top right.
3. **Dataset** — pick a JSONL chat file, CSV, folder of `.txt` files, or paste a Hugging Face Hub dataset id.
4. **Train** — review selections, tweak hyperparams, hit **Start training**.
5. **Runs** — list of all runs. Click one to see the live loss chart + log tail streaming over SSE.

---

## Run sidecar without the UI

For curl-based testing or scripting:

```bash
source .venv/bin/activate
uvicorn llm_chain_sidecar.main:app --host 127.0.0.1 --port 8000

# in another shell:
curl localhost:8000/api/hardware | python -m json.tool
curl 'localhost:8000/api/models?max_params=2000000000' | python -m json.tool
curl -X POST localhost:8000/api/runs \
  -H content-type:application/json \
  -d '{"model_id":"Qwen/Qwen3-0.6B","backend":"mlx","technique":"qlora","dataset_path":"/path/to/data.jsonl","epochs":1}'
```

## Probe hardware from Python (no server)

```bash
python -c "
from llm_chain_sidecar.hardware import probe_hardware
from llm_chain_sidecar.hardware.capabilities import capabilities_for_vram
report = probe_hardware()
for d in report.devices:
    cap = capabilities_for_vram(d.vram_gb, d.memory_kind)
    print(f'{d.name} ({d.vram_gb} GB, {d.memory_kind}) -> QLoRA up to {cap.qlora_max_params/1e9:.1f}B')
"
```

---

## GitHub Actions

Two workflows under `.github/workflows/`. Watch them at
https://github.com/Drake0306/LLM-Chain/actions.

### `ci.yml` — every push to `main`, every PR

Matrix runs `pytest` on `ubuntu-latest`, `macos-14`, `windows-latest`. Slow real-training tests are excluded by default (fast + free).

### `release.yml` — `v*` tag pushes (or manual `workflow_dispatch`)

| Matrix OS | Target | Sidecar extra | Bundle |
| --- | --- | --- | --- |
| `macos-14` | `aarch64-apple-darwin` | `mlx` | `.dmg` |
| `windows-latest` | `x86_64-pc-windows-msvc` | `cuda` | `.msi` (+ `.nsis`) |

Each leg installs the sidecar with its platform extras, runs `pytest`, runs `scripts/build-sidecar.sh` to produce a real PyInstaller binary, then runs `npm run tauri build`. Artifacts upload as `llm-chain-<target>` on the run page.

### Cut a release

```bash
git tag -a v0.1.1-alpha -m "release notes here"
git push origin v0.1.1-alpha
```

The first alpha is already pushed: tag [`v0.1.0-alpha`](https://github.com/Drake0306/LLM-Chain/releases/tag/v0.1.0-alpha).

To trigger the release workflow without a new tag:

```bash
gh workflow run release.yml --ref main
```

### Run CI checks locally before pushing

```bash
# what ci.yml runs:
cd sidecar && pytest -v

# what release.yml runs (slow — produces a real installer):
./scripts/build-sidecar.sh                  # without --dev
cd apps/desktop && npm run tauri build
# outputs land under apps/desktop/src-tauri/target/<triple>/release/bundle/
```

---

## Troubleshooting

**`zsh: no matches found: ./sidecar[dev]`** — quote the path: `pip install -e './sidecar[dev]'`. Zsh treats unquoted brackets as a glob.

**`resource path 'binaries/llm-chain-sidecar-...' doesn't exist`** during `npm run tauri dev` or `cargo check` — Tauri requires the sidecar binary to exist at build time. Run `./scripts/build-sidecar.sh --dev` (step 4 above).

**Window opens but the Dashboard says "Probing hardware…" forever** — the React UI is waiting for the sidecar port. Open devtools (Cmd+Opt+I on macOS) and check the console; usually means the sidecar process crashed at startup. Look at the Tauri terminal output for the sidecar's stderr.

**`Permission to ... denied to <other-account>`** when pushing — multi-GitHub-account SSH key collision. Either set up an SSH host alias in `~/.ssh/config` for the right account, or change the remote to HTTPS and use a personal access token: `git remote set-url origin https://github.com/Drake0306/LLM-Chain.git`.

---

## Where things live

```
apps/desktop/         Tauri 2 shell + React UI
  src/api/            Typed sidecar client + sidecar-port hook
  src/screens/        Dashboard, ModelPicker, DatasetPicker, Train, Runs
  src/state/          Selection context
  src-tauri/          Rust shell, capabilities, sidecar wiring
sidecar/              Python sidecar (FastAPI + ML)
  llm_chain_sidecar/api/          FastAPI routes (incl. SSE stream)
  llm_chain_sidecar/hardware/     Probe + VRAM-tier capability gating
  llm_chain_sidecar/models/       Apache/MIT allowlist registry
  llm_chain_sidecar/datasets/     JSONL/CSV/text-dir/HF loaders
  llm_chain_sidecar/runs/         Run store + executor
  llm_chain_sidecar/trainers/     HF/CUDA + MLX backends
scripts/              Sidecar build scripts (PyInstaller + dev wrapper)
docs/                 Dev + hardware docs + plans
.github/workflows/    CI (ci.yml) + release (release.yml)
```
