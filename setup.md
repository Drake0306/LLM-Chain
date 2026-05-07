# LLM-Chain — install + run

Repo: https://github.com/Drake0306/LLM-Chain
Latest release: https://github.com/Drake0306/LLM-Chain/releases/latest

Open-source desktop app to fine-tune an LLM on your own machine. Tauri 2 shell + Python sidecar; LoRA / QLoRA on NVIDIA CUDA and Apple Silicon MLX. AMD ROCm is detected and surfaced as **experimental** but the trainer is a stub — see the AMD ROCm section in step 2 below.

For deeper dev docs see [`docs/development.md`](docs/development.md). For the VRAM-tier table see [`docs/supported-hardware.md`](docs/supported-hardware.md).

---

## Install (end users)

Download from the [latest release](https://github.com/Drake0306/LLM-Chain/releases/latest).

### macOS (Apple Silicon)

1. Download `LLM-Chain_<version>_aarch64.dmg`.
2. Open the DMG → drag **LLM-Chain.app** to **Applications**.
3. **First launch:** macOS Gatekeeper will warn that the app is from an unidentified developer (binaries are unsigned in the alpha). Right-click the app → **Open** → confirm. Subsequent launches won't ask.
4. **Pin to the Dock:** with the app running, right-click its Dock icon → **Options** → **Keep in Dock**.

### Windows (x64)

1. Download `LLM-Chain_<version>_x64_en-US.msi` (or the `_x64-setup.exe` NSIS installer — both ship the same app).
2. Run it. SmartScreen will warn (unsigned binary in the alpha) — click **More info** → **Run anyway**.
3. After install, find **LLM-Chain** in the Start menu.

The first launch on either OS spawns the Python sidecar; you'll see the Dashboard once it has probed your hardware (~2–5 seconds).

---

## How to run it (from source)

> Both **macOS / Linux (bash/zsh)** and **Windows (PowerShell)** are supported. Where the commands differ, both are listed.

### 0. Prereqs

- **Python 3.11.x** (the sidecar pins `>=3.11,<3.12`)
- **Node 20+** and **npm**
- **Rust stable** (`rustup default stable`)
- **macOS arm64**, **Linux x86_64**, or **Windows x86_64**
- **Windows only:** Visual Studio 2022 Build Tools with the *Desktop development with C++* workload (Tauri's Rust shell needs MSVC + the Windows 10/11 SDK). Install from <https://visualstudio.microsoft.com/visual-cpp-build-tools/> and reboot once after install.

Check with:

**macOS / Linux:**
```bash
python3.11 --version    # should print 3.11.x
node --version          # should print v20+
rustc --version
```

**Windows (PowerShell):**
```powershell
python --version        # should print 3.11.x; if it points at 3.12+, install 3.11 from python.org and use `py -3.11` below
node --version          # should print v20+
rustc --version
```

### 1. Clone

```bash
git clone https://github.com/Drake0306/LLM-Chain.git
cd LLM-Chain
```

### 2. Set up the Python sidecar

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e './sidecar[dev]'
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".\sidecar[dev]"
```

If `Activate.ps1` fails with an execution-policy error, run once: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`. Quote `'./sidecar[dev]'` on macOS/Linux — zsh treats unquoted brackets as a glob.

To actually train (rather than just run the UI shell), also install the platform extra for your hardware:

**macOS / Linux:**
```bash
pip install -e './sidecar[dev,mlx]'         # macOS Apple Silicon
pip install -e './sidecar[dev,cuda]'        # NVIDIA Linux
```

**Windows (PowerShell):**
```powershell
pip install -e ".\sidecar[dev,cuda]"        # NVIDIA Windows
```

**AMD ROCm (experimental — Linux / WSL2 only):**

The Dashboard renders an amber "experimental — not yet validated on hardware" chip on detected AMD GPUs and the trainer refuses to instantiate; if you want to help us validate it, install a ROCm build of PyTorch (skip the `[cuda]` extra — `bitsandbytes` is CUDA-only):

```bash
pip install -e './sidecar[dev]'
pip install --pre torch --index-url https://download.pytorch.org/whl/rocm6.2
```

- **Linux:** native ROCm; supported AMD GPUs are listed at <https://rocm.docs.amd.com>.
- **Windows:** there is no native PyTorch+ROCm build for Windows. Use **WSL2 (Ubuntu 22.04 / 24.04)** and run the Linux instructions inside the WSL shell. The Microsoft DirectML stack (`torch-directml`) is a different code path that we do **not** detect.
- The probe keys off `torch.version.hip`, so a CUDA-built torch on an AMD-only box will leave the GPU invisible.

Please open an issue at <https://github.com/Drake0306/LLM-Chain/issues> with what you tried, what worked, and what didn't — that's how we get this off "experimental".

### 3. Verify the sidecar

**macOS / Linux:**
```bash
cd sidecar && pytest -v && cd ..
```

**Windows (PowerShell):**
```powershell
cd sidecar; pytest -v; cd ..
```

You should see most tests passing and a handful **deselected** — those are the slow real-training tests, gated behind `-m slow`.

### 4. Build the sidecar binary the Tauri shell expects

**macOS / Linux:**
```bash
./scripts/build-sidecar.sh --dev
```

**Windows (PowerShell):**
```powershell
.\scripts\build-sidecar.ps1 --dev
```

This writes a thin wrapper at `apps/desktop/src-tauri/binaries/llm-chain-sidecar-<your-triple>` that re-execs `python -m llm_chain_sidecar.main` from your `.venv`. Fast (~1 second) and only valid on this machine. For a portable binary, omit `--dev` — that runs PyInstaller and takes ~10 minutes.

### 5. Install desktop dependencies

**macOS / Linux:**
```bash
cd apps/desktop && npm install && cd ..
```

**Windows (PowerShell):**
```powershell
cd apps\desktop; npm install; cd ..\..
```

### 6. Launch the app

**macOS / Linux:**
```bash
cd apps/desktop && npm run tauri dev
```

**Windows (PowerShell):**
```powershell
cd apps\desktop; npm run tauri dev
```

The Tauri window opens, spawns the sidecar, parses its port off stdout, and the React UI hits `http://127.0.0.1:<port>`. First launch compiles the Rust shell (~1 minute); reloads after that are instant.

### 7. Click through the flow

1. **Dashboard** — your hardware report. Pick a device (CUDA or MLX); CPU is selectable for ≤100M models. AMD GPUs (ROCm) appear with an amber "experimental — not yet validated on hardware" chip and are not selectable yet — see the AMD ROCm note above.
2. **Model** — picks from the curated Apache/MIT allowlist, gated by your selected device's QLoRA cap. Toggle QLoRA / LoRA at the top right.
3. **Dataset** — pick a JSONL chat file, CSV, folder of `.txt` files, or paste a Hugging Face Hub dataset id.
4. **Train** — review selections, tweak hyperparams, hit **Start training**.
5. **Runs** — list of all runs. Click one to see the live loss chart + log tail streaming over SSE. While weights are downloading, a progress bar replaces the chart placeholder. Use **Cancel** to stop a run mid-step (the trainer marks it `canceled` and writes nothing further). Use **Reveal in Finder/Explorer** to jump straight to the saved adapter directory.
6. **Settings** — defaults applied to new runs: preferred backend (auto / cuda / mlx), default dataset format, output directory override. Backend / format prefs apply instantly; output dir takes effect on the next app launch (the sidecar reads `LLM_CHAIN_RUNS_DIR` only at startup).

A high-level diagram of how data flows between the UI, sidecar, executor, and trainers lives in [`docs/data-flow.md`](docs/data-flow.md).

---

## Run sidecar without the UI

For curl-based testing or scripting.

**macOS / Linux:**
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

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
uvicorn llm_chain_sidecar.main:app --host 127.0.0.1 --port 8000

# in another shell:
Invoke-RestMethod http://localhost:8000/api/hardware | ConvertTo-Json -Depth 5
Invoke-RestMethod 'http://localhost:8000/api/models?max_params=2000000000' | ConvertTo-Json -Depth 5
$body = @{
    model_id     = "Qwen/Qwen3-0.6B"
    backend      = "cuda"
    technique    = "qlora"
    dataset_path = "C:\path\to\data.jsonl"
    epochs       = 1
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/runs -Body $body -ContentType application/json
```

## Probe hardware from Python (no server)

**macOS / Linux:**
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

**Windows (PowerShell):**
```powershell
@"
from llm_chain_sidecar.hardware import probe_hardware
from llm_chain_sidecar.hardware.capabilities import capabilities_for_vram
report = probe_hardware()
for d in report.devices:
    cap = capabilities_for_vram(d.vram_gb, d.memory_kind)
    print(f'{d.name} ({d.vram_gb} GB, {d.memory_kind}) -> QLoRA up to {cap.qlora_max_params/1e9:.1f}B')
"@ | python -
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

**Export GGUF says `convert_hf_to_gguf.py not found`** — you need llama.cpp tooling.

- **macOS / Linux:** `./scripts/llama-cpp-bootstrap.sh` clones llama.cpp into `~/.llm-chain/llama.cpp` and builds the `llama-quantize` binary that k-quants like `q4_k_m` need.
- **Windows (PowerShell):** the bootstrap script is bash-only. Easiest path: download a precompiled llama.cpp release from <https://github.com/ggerganov/llama.cpp/releases> (the Windows AVX2 zip), extract it to `%USERPROFILE%\.llm-chain\llama.cpp\`, and `pip install gguf>=0.9` in the venv. Set `LLAMA_CPP_DIR` env var if you put it elsewhere. `f16` and `q8_0` work without `llama-quantize.exe`; only k-quants need it.
- Either platform: skip GGUF entirely if you just want the merged HF directory — the export panel surfaces "Reveal merged model" on convert failure, and the merged dir loads directly with `mlx_lm.generate` or `transformers.from_pretrained`.

**Hugging Face token (silences the rate-limit warning, enables Hub push)** — set before `npm run tauri dev` so the sidecar inherits it.

- **macOS / Linux:** `huggingface-cli login` (writes `~/.cache/huggingface/token`) or `export HF_TOKEN="hf_..."`.
- **Windows (PowerShell):** `huggingface-cli login` (writes `%USERPROFILE%\.cache\huggingface\token`) or `$env:HF_TOKEN = "hf_..."`. Restart the app afterwards — the token is read once at sidecar startup.

---

## Where things live

```
apps/desktop/         Tauri 2 shell + React UI
  src/api/            Typed sidecar client + sidecar-port hook
  src/screens/        Dashboard, ModelPicker, DatasetPicker, Train, Runs, Settings
  src/state/          Selection context + persisted Settings
  src-tauri/          Rust shell, capabilities, sidecar wiring (reads desktop-settings.json)
sidecar/              Python sidecar (FastAPI + ML)
  llm_chain_sidecar/api/          FastAPI routes (incl. SSE stream + cancel)
  llm_chain_sidecar/hardware/     Probe + VRAM-tier capability gating
  llm_chain_sidecar/models/       Apache/MIT allowlist registry
  llm_chain_sidecar/datasets/     JSONL/CSV/text-dir/HF loaders
  llm_chain_sidecar/runs/         Run store + executor (with cancellation tokens)
  llm_chain_sidecar/trainers/     HF/CUDA + MLX backends + tqdm download bridge
  llm_chain_sidecar/exports/      GGUF export (peft merge + llama.cpp convert)
scripts/              Sidecar build scripts (PyInstaller + dev wrapper) + llama-cpp bootstrap
docs/                 Dev + hardware docs + data-flow + plans
.github/workflows/    CI (ci.yml) + release (release.yml)
```
