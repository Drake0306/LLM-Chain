# Local LLM Trainer v1.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the foundation of a cross-platform desktop app where a user can pick a base model, pick a dataset, and run a LoRA fine-tune locally on NVIDIA (CUDA) or Apple Silicon (MLX) hardware — with the UI gating choices to what their hardware can handle.

**Architecture:** Tauri 2.x desktop shell (Rust + React/TypeScript frontend) drives a Python sidecar (FastAPI on localhost) that performs all hardware probing, dataset loading, and training. Sidecar processes are spawned as Tauri `externalBin`. UI ↔ sidecar communication is JSON over HTTP for commands and Server-Sent Events for streaming training logs/metrics.

**Tech Stack:**
- Shell: Tauri 2.x, Rust
- Frontend: React 18 + TypeScript + Vite + Tailwind CSS + Recharts
- Sidecar: Python 3.11, FastAPI, uvicorn
- ML: PyTorch, Hugging Face `transformers` + `trl` + `peft` + `accelerate`, MLX + `mlx-lm` (Apple Silicon)
- Datasets: `datasets` (HF), pandas (CSV)
- Packaging: `python-build-standalone` for relocatable Python; PyInstaller fallback
- Testing: pytest (Python), vitest (TS), `cargo test` (Rust), Playwright (e2e UI smoke)

**Scope of v1.0** (other backends/features land in v1.1 and v1.2):
- Backends: NVIDIA CUDA + Apple Silicon MLX only (AMD/Intel/CPU defer)
- Models: Apache/MIT only from curated allowlist (no restricted-license toggle yet)
- Training mode: LoRA + QLoRA fine-tuning only (no pretraining-from-scratch, no full FT, no multimodal)
- Datasets: JSONL chat format, CSV, plain-text directory, HF datasets ID (no images)
- Export: local checkpoints only (no GGUF, no HF Hub push)
- Telemetry: none (defer to v1.1)

---

## Repository Layout

```
LLM-Chain/
├── apps/
│   └── desktop/                 # Tauri shell + React frontend
│       ├── src/                 # React/TS
│       ├── src-tauri/           # Rust shell
│       └── package.json
├── sidecar/                     # Python ML sidecar
│   ├── llm_chain_sidecar/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app entrypoint
│   │   ├── hardware/            # GPU/RAM probing
│   │   ├── models/              # Model registry + loaders
│   │   ├── datasets/            # Dataset loaders
│   │   ├── trainers/            # CUDA + MLX trainer dispatch
│   │   ├── runs/                # Run manager
│   │   └── api/                 # FastAPI routes
│   ├── tests/
│   └── pyproject.toml
├── docs/
│   └── plans/
├── .github/workflows/           # CI for Mac + Windows
├── LICENSE                      # Apache 2.0
├── README.md
└── .gitignore
```

---

## Task 1: Project scaffolding + license + CI skeleton

**Files:**
- Create: `LICENSE`, `README.md`, `.gitignore`, `.editorconfig`
- Create: `.github/workflows/ci.yml`

**Step 1: Create the Apache 2.0 LICENSE file**

Use the standard Apache 2.0 text from `https://www.apache.org/licenses/LICENSE-2.0.txt`. Fill in `[yyyy]` = 2026 and `[name of copyright owner]` = "LLM-Chain contributors".

**Step 2: Write README.md**

```markdown
# LLM-Chain

Open-source desktop app to train your own LLM locally on your own hardware.

**Status:** v1.0 in development.

**Supported hardware (v1.0):** NVIDIA CUDA (Linux + Windows), Apple Silicon (macOS).
**Coming in v1.1:** AMD ROCm, Intel XPU, CPU fallback.
**Coming in v1.2:** Pretraining-from-scratch, multimodal (image+text).

## License
Apache 2.0. See LICENSE.

## Building from source
See `docs/development.md`.
```

**Step 3: Write .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/

# Node
node_modules/
dist/
.vite/

# Rust / Tauri
target/
src-tauri/target/

# OS
.DS_Store
Thumbs.db

# Local data
~/.llm-chain/
runs/
checkpoints/
*.safetensors
*.bin

# Editor
.idea/
.vscode/
*.swp
```

**Step 4: Write .editorconfig**

```ini
root = true

[*]
end_of_line = lf
insert_final_newline = true
charset = utf-8
indent_style = space
trim_trailing_whitespace = true

[*.{py}]
indent_size = 4

[*.{ts,tsx,js,jsx,json,yaml,yml,md,toml}]
indent_size = 2

[*.rs]
indent_size = 4

[Makefile]
indent_style = tab
```

**Step 5: Write minimal CI skeleton at `.github/workflows/ci.yml`**

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  python:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-14, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -e ./sidecar[dev]
      - run: pytest sidecar/tests -v
```

**Step 6: Commit**

```bash
git add LICENSE README.md .gitignore .editorconfig .github/workflows/ci.yml
git commit -m "chore: scaffold repo (license, readme, ci skeleton)"
```

---

## Task 2: Python sidecar — package + FastAPI hello

**Files:**
- Create: `sidecar/pyproject.toml`
- Create: `sidecar/llm_chain_sidecar/__init__.py`
- Create: `sidecar/llm_chain_sidecar/main.py`
- Create: `sidecar/tests/__init__.py`
- Create: `sidecar/tests/test_main.py`

**Step 1: Write the failing test at `sidecar/tests/test_main.py`**

```python
from fastapi.testclient import TestClient
from llm_chain_sidecar.main import app


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}
```

**Step 2: Write `sidecar/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-chain-sidecar"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "pydantic>=2.6",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
    "ruff>=0.4",
    "mypy>=1.10",
]

[project.scripts]
llm-chain-sidecar = "llm_chain_sidecar.main:run"

[tool.setuptools.packages.find]
include = ["llm_chain_sidecar*"]
```

**Step 3: Run test to verify it fails**

Run: `cd sidecar && pip install -e .[dev] && pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_chain_sidecar.main'`

**Step 4: Write minimal `sidecar/llm_chain_sidecar/__init__.py`**

```python
__version__ = "0.1.0"
```

**Step 5: Write minimal `sidecar/llm_chain_sidecar/main.py`**

```python
from fastapi import FastAPI
import uvicorn
from . import __version__

app = FastAPI(title="LLM-Chain Sidecar", version=__version__)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


def run() -> None:
    uvicorn.run(app, host="127.0.0.1", port=0)
```

**Step 6: Run test to verify it passes**

Run: `pytest sidecar/tests/test_main.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add sidecar/
git commit -m "feat(sidecar): scaffold FastAPI app with /health endpoint"
```

---

## Task 3: Hardware probe — data model + dispatcher

**Files:**
- Create: `sidecar/llm_chain_sidecar/hardware/__init__.py`
- Create: `sidecar/llm_chain_sidecar/hardware/types.py`
- Create: `sidecar/llm_chain_sidecar/hardware/probe.py`
- Create: `sidecar/tests/hardware/test_probe.py`

**Step 1: Write the failing test at `sidecar/tests/hardware/test_probe.py`**

```python
from llm_chain_sidecar.hardware.probe import probe_hardware
from llm_chain_sidecar.hardware.types import HardwareReport, Backend


def test_probe_returns_report_with_at_least_cpu():
    report = probe_hardware()
    assert isinstance(report, HardwareReport)
    assert report.os in ("Darwin", "Windows", "Linux")
    assert report.cpu.cores >= 1
    assert report.system_ram_gb > 0
    # CPU is always present as a fallback backend
    assert any(d.backend == Backend.CPU for d in report.devices)


def test_probe_is_idempotent():
    a = probe_hardware()
    b = probe_hardware()
    assert a.os == b.os
    assert a.cpu.cores == b.cpu.cores
    assert len(a.devices) == len(b.devices)
```

**Step 2: Write `sidecar/llm_chain_sidecar/hardware/types.py`**

```python
from enum import Enum
from pydantic import BaseModel


class Backend(str, Enum):
    CUDA = "cuda"
    MPS = "mps"      # Apple Silicon via PyTorch MPS
    MLX = "mlx"      # Apple Silicon via MLX framework
    ROCM = "rocm"    # AMD (v1.1)
    XPU = "xpu"      # Intel (v1.1)
    CPU = "cpu"


class CpuInfo(BaseModel):
    cores: int
    name: str


class GpuDevice(BaseModel):
    backend: Backend
    name: str
    vram_gb: float          # 0 for CPU; for Apple Silicon = unified memory pool
    is_unified_memory: bool # True only for Apple Silicon
    driver_version: str | None = None


class HardwareReport(BaseModel):
    os: str                 # "Darwin", "Windows", "Linux"
    os_version: str
    cpu: CpuInfo
    system_ram_gb: float
    devices: list[GpuDevice]
```

**Step 3: Write `sidecar/llm_chain_sidecar/hardware/probe.py`**

```python
import platform
import psutil
from .types import Backend, CpuInfo, GpuDevice, HardwareReport


def probe_hardware() -> HardwareReport:
    devices: list[GpuDevice] = []
    devices.extend(_probe_cuda())
    devices.extend(_probe_apple())
    devices.append(GpuDevice(
        backend=Backend.CPU, name="CPU",
        vram_gb=0.0, is_unified_memory=False,
    ))

    return HardwareReport(
        os=platform.system(),
        os_version=platform.release(),
        cpu=CpuInfo(cores=psutil.cpu_count(logical=False) or 1, name=platform.processor() or "unknown"),
        system_ram_gb=round(psutil.virtual_memory().total / (1024**3), 2),
        devices=devices,
    )


def _probe_cuda() -> list[GpuDevice]:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            out.append(GpuDevice(
                backend=Backend.CUDA,
                name=props.name,
                vram_gb=round(props.total_memory / (1024**3), 2),
                is_unified_memory=False,
                driver_version=getattr(torch.version, "cuda", None),
            ))
        return out
    except Exception:
        return []


def _probe_apple() -> list[GpuDevice]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    # Apple Silicon: unified memory == system RAM
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    devices = [GpuDevice(
        backend=Backend.MLX,
        name="Apple Silicon GPU (MLX)",
        vram_gb=ram_gb,
        is_unified_memory=True,
    )]
    try:
        import torch
        if torch.backends.mps.is_available():
            devices.append(GpuDevice(
                backend=Backend.MPS,
                name="Apple Silicon GPU (PyTorch MPS)",
                vram_gb=ram_gb,
                is_unified_memory=True,
            ))
    except Exception:
        pass
    return devices
```

**Step 4: Add `psutil` and `torch` to sidecar deps**

Edit `sidecar/pyproject.toml` `dependencies`:

```toml
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "pydantic>=2.6",
    "psutil>=5.9",
    "torch>=2.3",
]
```

Reinstall: `pip install -e ./sidecar[dev]`

**Step 5: Write `sidecar/llm_chain_sidecar/hardware/__init__.py`**

```python
from .probe import probe_hardware
from .types import Backend, GpuDevice, HardwareReport, CpuInfo

__all__ = ["probe_hardware", "Backend", "GpuDevice", "HardwareReport", "CpuInfo"]
```

**Step 6: Run tests**

Create `sidecar/tests/hardware/__init__.py` (empty file).

Run: `pytest sidecar/tests/hardware -v`
Expected: PASS on every OS

**Step 7: Commit**

```bash
git add sidecar/llm_chain_sidecar/hardware/ sidecar/tests/hardware/ sidecar/pyproject.toml
git commit -m "feat(hardware): probe CPU/RAM and detect CUDA + Apple Silicon devices"
```

---

## Task 4: VRAM-tier capability gating

**Files:**
- Create: `sidecar/llm_chain_sidecar/hardware/capabilities.py`
- Create: `sidecar/tests/hardware/test_capabilities.py`

**Step 1: Write the failing test**

```python
from llm_chain_sidecar.hardware.capabilities import (
    Capability, capabilities_for_vram, MAX_PARAMS_BY_TIER,
)


def test_8gb_can_qlora_7b_but_not_full_ft_above_125m():
    caps = capabilities_for_vram(8.0)
    assert caps.qlora_max_params >= 7_000_000_000
    assert caps.lora_max_params <= 3_000_000_000
    assert caps.full_ft_max_params <= 125_000_000


def test_24gb_can_qlora_30b():
    caps = capabilities_for_vram(24.0)
    assert caps.qlora_max_params >= 13_000_000_000
    assert caps.full_ft_max_params >= 1_000_000_000


def test_128gb_unified_can_qlora_70b():
    caps = capabilities_for_vram(128.0, is_unified_memory=True)
    assert caps.qlora_max_params >= 70_000_000_000


def test_pc_shared_memory_is_treated_as_zero():
    # Under 8GB shouldn't pretend to access "shared" memory
    caps = capabilities_for_vram(4.0)
    assert caps.qlora_max_params < 7_000_000_000
```

**Step 2: Write `sidecar/llm_chain_sidecar/hardware/capabilities.py`**

```python
from dataclasses import dataclass

# Tier table from design doc Section 4 (VRAM heuristics, 2026)
# Each entry: (min_vram_gb, qlora_max, lora_max, full_ft_max)
_TIERS = [
    (8.0,   7_000_000_000,  3_000_000_000,    125_000_000),
    (12.0,  13_000_000_000, 7_000_000_000,    350_000_000),
    (16.0,  13_000_000_000, 13_000_000_000,   500_000_000),
    (24.0,  34_000_000_000, 13_000_000_000,   1_300_000_000),
    (32.0,  30_000_000_000, 13_000_000_000,   1_500_000_000),
    (48.0,  70_000_000_000, 30_000_000_000,   3_000_000_000),
    (64.0,  70_000_000_000, 30_000_000_000,   3_000_000_000),
    (128.0, 70_000_000_000, 70_000_000_000,   7_000_000_000),
]

MAX_PARAMS_BY_TIER = _TIERS  # exported for the UI


@dataclass(frozen=True)
class Capability:
    qlora_max_params: int
    lora_max_params: int
    full_ft_max_params: int
    notes: str


def capabilities_for_vram(vram_gb: float, is_unified_memory: bool = False) -> Capability:
    """Return the max trainable params at each technique for a given VRAM tier.

    On Apple Silicon, unified memory really is GPU-accessible. On PC, "shared GPU
    memory" is slow PCIe spillover and must NOT be counted toward VRAM here.
    """
    if vram_gb < _TIERS[0][0]:
        return Capability(
            qlora_max_params=1_000_000_000,
            lora_max_params=350_000_000,
            full_ft_max_params=50_000_000,
            notes="Below 8 GB — only very small models / tiny LoRAs.",
        )
    chosen = _TIERS[0]
    for tier in _TIERS:
        if vram_gb >= tier[0]:
            chosen = tier
    note = "Apple unified memory — ~75% addressable for GPU." if is_unified_memory else ""
    return Capability(
        qlora_max_params=chosen[1],
        lora_max_params=chosen[2],
        full_ft_max_params=chosen[3],
        notes=note,
    )
```

**Step 3: Run tests**

Run: `pytest sidecar/tests/hardware/test_capabilities.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add sidecar/llm_chain_sidecar/hardware/capabilities.py sidecar/tests/hardware/test_capabilities.py
git commit -m "feat(hardware): VRAM-tier capability gating for QLoRA/LoRA/full-FT"
```

---

## Task 5: Model registry — curated allowlist

**Files:**
- Create: `sidecar/llm_chain_sidecar/models/__init__.py`
- Create: `sidecar/llm_chain_sidecar/models/registry.py`
- Create: `sidecar/llm_chain_sidecar/models/data/allowlist.yaml`
- Create: `sidecar/tests/models/test_registry.py`

**Step 1: Write the failing test**

```python
from llm_chain_sidecar.models.registry import ModelRegistry, ModelEntry, License


def test_registry_loads_allowlist():
    reg = ModelRegistry.load_default()
    assert len(reg.entries) > 5
    assert all(isinstance(e, ModelEntry) for e in reg.entries)


def test_default_excludes_restricted_licenses():
    reg = ModelRegistry.load_default()
    assert all(e.license in (License.APACHE_2_0, License.MIT) for e in reg.entries)


def test_filter_by_max_params():
    reg = ModelRegistry.load_default()
    small = reg.fitting_within(500_000_000)
    assert all(e.params <= 500_000_000 for e in small)
    assert any("Pythia" in e.name or "SmolLM" in e.name for e in small)
```

**Step 2: Write `sidecar/llm_chain_sidecar/models/registry.py`**

```python
from enum import Enum
from pathlib import Path
import yaml
from pydantic import BaseModel


class License(str, Enum):
    APACHE_2_0 = "Apache-2.0"
    MIT = "MIT"
    LLAMA_COMMUNITY = "Llama-Community"
    GEMMA = "Gemma-ToU"
    FALCON = "Falcon-LLM-2.0"
    DEEPSEEK = "DeepSeek-Model-License"
    NVIDIA_OPEN = "NVIDIA-Open-Model"


class ModelEntry(BaseModel):
    id: str                  # HF Hub id, e.g. "Qwen/Qwen3-1.7B"
    name: str                # Display name
    family: str              # "Qwen3", "Mistral", ...
    params: int              # parameter count
    license: License
    license_caveat: str | None = None
    modalities: list[str]    # ["text"], ["text", "image"]
    supports_lora: bool = True
    notes: str | None = None


class ModelRegistry(BaseModel):
    entries: list[ModelEntry]

    @classmethod
    def load_default(cls) -> "ModelRegistry":
        path = Path(__file__).parent / "data" / "allowlist.yaml"
        return cls.from_yaml(path)

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelRegistry":
        with path.open() as f:
            raw = yaml.safe_load(f)
        return cls(entries=[ModelEntry(**e) for e in raw["models"]])

    def fitting_within(self, max_params: int) -> list[ModelEntry]:
        return [e for e in self.entries if e.params <= max_params]
```

**Step 3: Write `sidecar/llm_chain_sidecar/models/data/allowlist.yaml`**

Curated from design doc Section 5. v1.0 = Apache/MIT only, text-only.

```yaml
models:
  # ---------- Small (10M-500M) ----------
  - id: EleutherAI/pythia-70m
    name: Pythia 70M
    family: Pythia
    params: 70_000_000
    license: Apache-2.0
    modalities: [text]
    notes: Open data, great from-scratch reference.
  - id: EleutherAI/pythia-410m
    name: Pythia 410M
    family: Pythia
    params: 410_000_000
    license: Apache-2.0
    modalities: [text]
  - id: HuggingFaceTB/SmolLM2-360M
    name: SmolLM2 360M
    family: SmolLM
    params: 360_000_000
    license: Apache-2.0
    modalities: [text]
  - id: Qwen/Qwen3-0.6B
    name: Qwen3 0.6B
    family: Qwen3
    params: 600_000_000
    license: Apache-2.0
    modalities: [text]
  - id: microsoft/Phi-4-mini-instruct
    name: Phi-4 mini instruct
    family: Phi
    params: 3_800_000_000
    license: MIT
    modalities: [text]

  # ---------- Mid (1B-8B) ----------
  - id: Qwen/Qwen3-1.7B
    name: Qwen3 1.7B
    family: Qwen3
    params: 1_700_000_000
    license: Apache-2.0
    modalities: [text]
  - id: Qwen/Qwen3-4B
    name: Qwen3 4B
    family: Qwen3
    params: 4_000_000_000
    license: Apache-2.0
    modalities: [text]
  - id: Qwen/Qwen3-8B
    name: Qwen3 8B
    family: Qwen3
    params: 8_000_000_000
    license: Apache-2.0
    modalities: [text]
  - id: mistralai/Mistral-7B-v0.3
    name: Mistral 7B v0.3
    family: Mistral
    params: 7_000_000_000
    license: Apache-2.0
    modalities: [text]
  - id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    name: TinyLlama 1.1B Chat
    family: TinyLlama
    params: 1_100_000_000
    license: Apache-2.0
    modalities: [text]

  # ---------- Large (13B+) ----------
  - id: Qwen/Qwen3-32B
    name: Qwen3 32B
    family: Qwen3
    params: 32_000_000_000
    license: Apache-2.0
    modalities: [text]
  - id: allenai/OLMo-2-1124-13B
    name: OLMo 2 13B
    family: OLMo
    params: 13_000_000_000
    license: Apache-2.0
    modalities: [text]
    notes: Card says research/educational; weights are Apache-2.0.
```

**Step 4: Write `sidecar/llm_chain_sidecar/models/__init__.py`**

```python
from .registry import ModelEntry, ModelRegistry, License

__all__ = ["ModelEntry", "ModelRegistry", "License"]
```

**Step 5: Add `pyyaml` to sidecar deps**

Add to `dependencies` in `sidecar/pyproject.toml`: `"pyyaml>=6.0"`. Reinstall.

**Step 6: Run tests**

Create `sidecar/tests/models/__init__.py` (empty).
Run: `pytest sidecar/tests/models -v`
Expected: PASS

**Step 7: Commit**

```bash
git add sidecar/llm_chain_sidecar/models/ sidecar/tests/models/ sidecar/pyproject.toml
git commit -m "feat(models): curated Apache/MIT model allowlist registry"
```

---

## Task 6: Dataset loader — JSONL chat format

**Files:**
- Create: `sidecar/llm_chain_sidecar/datasets/__init__.py`
- Create: `sidecar/llm_chain_sidecar/datasets/types.py`
- Create: `sidecar/llm_chain_sidecar/datasets/loader.py`
- Create: `sidecar/tests/datasets/test_loader_jsonl.py`

**Step 1: Write failing test**

```python
import json
from pathlib import Path
from llm_chain_sidecar.datasets.loader import load_dataset
from llm_chain_sidecar.datasets.types import DatasetSource, DatasetFormat


def test_jsonl_chat_loads(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"},
                                 {"role": "assistant", "content": "hello"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "bye"},
                                   {"role": "assistant", "content": "goodbye"}]}) + "\n"
    )
    src = DatasetSource(format=DatasetFormat.JSONL_CHAT, path=str(p))
    ds = load_dataset(src)
    assert len(ds) == 2
    assert ds[0]["messages"][0]["role"] == "user"


def test_jsonl_chat_rejects_malformed(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"not_messages": []}\n')
    src = DatasetSource(format=DatasetFormat.JSONL_CHAT, path=str(p))
    import pytest
    with pytest.raises(ValueError, match="missing 'messages'"):
        load_dataset(src)
```

**Step 2: Write `sidecar/llm_chain_sidecar/datasets/types.py`**

```python
from enum import Enum
from pydantic import BaseModel


class DatasetFormat(str, Enum):
    JSONL_CHAT = "jsonl_chat"   # {"messages": [{"role": ..., "content": ...}, ...]}
    CSV = "csv"                 # columns chosen by user
    TEXT_DIR = "text_dir"       # folder of .txt files
    HF_HUB = "hf_hub"           # HF datasets id


class DatasetSource(BaseModel):
    format: DatasetFormat
    path: str | None = None     # local path for JSONL/CSV/TEXT_DIR
    hf_id: str | None = None    # for HF_HUB
    split: str = "train"
    text_column: str | None = None  # for CSV
```

**Step 3: Write `sidecar/llm_chain_sidecar/datasets/loader.py` (JSONL only for now)**

```python
import json
from pathlib import Path
from .types import DatasetSource, DatasetFormat


def load_dataset(src: DatasetSource) -> list[dict]:
    if src.format == DatasetFormat.JSONL_CHAT:
        return _load_jsonl_chat(Path(src.path))
    raise NotImplementedError(f"Format {src.format} not yet implemented")


def _load_jsonl_chat(path: Path) -> list[dict]:
    rows: list[dict] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "messages" not in obj:
            raise ValueError(f"Row {i}: missing 'messages' key")
        if not isinstance(obj["messages"], list) or not obj["messages"]:
            raise ValueError(f"Row {i}: 'messages' must be a non-empty list")
        for j, m in enumerate(obj["messages"]):
            if "role" not in m or "content" not in m:
                raise ValueError(f"Row {i} msg {j}: missing role/content")
        rows.append(obj)
    return rows
```

**Step 4: Run tests**

Create `sidecar/tests/datasets/__init__.py` (empty).
Run: `pytest sidecar/tests/datasets -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sidecar/llm_chain_sidecar/datasets/ sidecar/tests/datasets/
git commit -m "feat(datasets): JSONL chat dataset loader with validation"
```

---

## Task 7: Dataset loader — CSV, TEXT_DIR, HF_HUB

**Files:**
- Modify: `sidecar/llm_chain_sidecar/datasets/loader.py`
- Create: `sidecar/tests/datasets/test_loader_csv.py`
- Create: `sidecar/tests/datasets/test_loader_textdir.py`
- Create: `sidecar/tests/datasets/test_loader_hf.py`

**Step 1: Write CSV test**

```python
from pathlib import Path
from llm_chain_sidecar.datasets.loader import load_dataset
from llm_chain_sidecar.datasets.types import DatasetSource, DatasetFormat


def test_csv_loads_with_text_column(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("text,label\nhello,greet\nbye,farewell\n")
    src = DatasetSource(format=DatasetFormat.CSV, path=str(p), text_column="text")
    ds = load_dataset(src)
    assert len(ds) == 2
    assert ds[0]["text"] == "hello"


def test_csv_requires_text_column(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("foo,bar\n1,2\n")
    src = DatasetSource(format=DatasetFormat.CSV, path=str(p), text_column="text")
    import pytest
    with pytest.raises(ValueError, match="column 'text' not found"):
        load_dataset(src)
```

**Step 2: Write text-dir test**

```python
from pathlib import Path
from llm_chain_sidecar.datasets.loader import load_dataset
from llm_chain_sidecar.datasets.types import DatasetSource, DatasetFormat


def test_text_dir_loads_all_txt_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("first")
    (tmp_path / "b.txt").write_text("second")
    (tmp_path / "skip.md").write_text("ignored")
    src = DatasetSource(format=DatasetFormat.TEXT_DIR, path=str(tmp_path))
    ds = load_dataset(src)
    assert len(ds) == 2
    assert {r["text"] for r in ds} == {"first", "second"}
```

**Step 3: Write HF Hub test (mocked, no network)**

```python
from unittest.mock import patch, MagicMock
from llm_chain_sidecar.datasets.loader import load_dataset
from llm_chain_sidecar.datasets.types import DatasetSource, DatasetFormat


def test_hf_hub_dispatches_to_datasets_library():
    fake_rows = [{"text": "x"}, {"text": "y"}]
    with patch("llm_chain_sidecar.datasets.loader._hf_load") as m:
        m.return_value = fake_rows
        src = DatasetSource(format=DatasetFormat.HF_HUB, hf_id="acme/dataset", split="train")
        ds = load_dataset(src)
        assert ds == fake_rows
        m.assert_called_once_with("acme/dataset", "train")
```

**Step 4: Update `loader.py` with CSV, TEXT_DIR, HF_HUB**

```python
import csv
import json
from pathlib import Path
from .types import DatasetSource, DatasetFormat


def load_dataset(src: DatasetSource) -> list[dict]:
    if src.format == DatasetFormat.JSONL_CHAT:
        return _load_jsonl_chat(Path(src.path))
    if src.format == DatasetFormat.CSV:
        return _load_csv(Path(src.path), src.text_column)
    if src.format == DatasetFormat.TEXT_DIR:
        return _load_text_dir(Path(src.path))
    if src.format == DatasetFormat.HF_HUB:
        return _hf_load(src.hf_id, src.split)
    raise NotImplementedError(f"Format {src.format} not implemented")


def _load_jsonl_chat(path: Path) -> list[dict]:
    rows: list[dict] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "messages" not in obj:
            raise ValueError(f"Row {i}: missing 'messages' key")
        if not isinstance(obj["messages"], list) or not obj["messages"]:
            raise ValueError(f"Row {i}: 'messages' must be a non-empty list")
        for j, m in enumerate(obj["messages"]):
            if "role" not in m or "content" not in m:
                raise ValueError(f"Row {i} msg {j}: missing role/content")
        rows.append(obj)
    return rows


def _load_csv(path: Path, text_column: str | None) -> list[dict]:
    if not text_column:
        raise ValueError("CSV format requires text_column")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if text_column not in (reader.fieldnames or []):
            raise ValueError(f"column '{text_column}' not found in CSV")
        return list(reader)


def _load_text_dir(path: Path) -> list[dict]:
    return [{"text": p.read_text()} for p in sorted(path.glob("*.txt"))]


def _hf_load(hf_id: str, split: str) -> list[dict]:
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset(hf_id, split=split)
    return [dict(row) for row in ds]
```

**Step 5: Add `datasets` to sidecar deps**

Add `"datasets>=2.18"` to `dependencies` in `pyproject.toml`. Reinstall.

**Step 6: Run all dataset tests**

Run: `pytest sidecar/tests/datasets -v`
Expected: 5 PASS (1 JSONL + 2 CSV + 1 text-dir + 1 HF mocked)

**Step 7: Commit**

```bash
git add sidecar/llm_chain_sidecar/datasets/ sidecar/tests/datasets/ sidecar/pyproject.toml
git commit -m "feat(datasets): add CSV, text-dir, HF Hub loaders"
```

---

## Task 8: Run manager — types + filesystem layout

**Files:**
- Create: `sidecar/llm_chain_sidecar/runs/__init__.py`
- Create: `sidecar/llm_chain_sidecar/runs/types.py`
- Create: `sidecar/llm_chain_sidecar/runs/store.py`
- Create: `sidecar/tests/runs/test_store.py`

**Step 1: Write failing test**

```python
from pathlib import Path
from llm_chain_sidecar.runs.store import RunStore
from llm_chain_sidecar.runs.types import RunConfig, RunStatus


def test_create_run_persists_config(tmp_path: Path):
    store = RunStore(root=tmp_path)
    cfg = RunConfig(
        model_id="Qwen/Qwen3-0.6B",
        backend="cuda",
        technique="lora",
        dataset_path="/tmp/x.jsonl",
        epochs=1,
    )
    run = store.create(cfg)
    assert run.id
    assert run.status == RunStatus.PENDING
    loaded = store.get(run.id)
    assert loaded.config.model_id == "Qwen/Qwen3-0.6B"


def test_list_runs_returns_newest_first(tmp_path: Path):
    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    a = store.create(cfg)
    b = store.create(cfg)
    runs = store.list()
    assert runs[0].id == b.id
    assert runs[1].id == a.id


def test_update_status(tmp_path: Path):
    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    store.update_status(run.id, RunStatus.RUNNING)
    assert store.get(run.id).status == RunStatus.RUNNING
```

**Step 2: Write `sidecar/llm_chain_sidecar/runs/types.py`**

```python
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field
from uuid import uuid4


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class RunConfig(BaseModel):
    model_id: str
    backend: str            # "cuda", "mlx", etc.
    technique: str          # "lora", "qlora"
    dataset_path: str
    dataset_format: str = "jsonl_chat"
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32


class Run(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: RunStatus = RunStatus.PENDING
    config: RunConfig
    error: str | None = None
    output_dir: str | None = None
```

**Step 3: Write `sidecar/llm_chain_sidecar/runs/store.py`**

```python
import json
from pathlib import Path
from .types import Run, RunConfig, RunStatus


class RunStore:
    """JSON-on-disk store at <root>/<run_id>/run.json."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, config: RunConfig) -> Run:
        run = Run(config=config)
        run.output_dir = str(self.root / run.id)
        Path(run.output_dir).mkdir(parents=True, exist_ok=True)
        self._write(run)
        return run

    def get(self, run_id: str) -> Run:
        return Run.model_validate_json((self.root / run_id / "run.json").read_text())

    def list(self) -> list[Run]:
        runs = [self.get(p.name) for p in self.root.iterdir() if (p / "run.json").exists()]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)

    def update_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        run = self.get(run_id)
        run.status = status
        if error:
            run.error = error
        self._write(run)

    def _write(self, run: Run) -> None:
        (Path(run.output_dir) / "run.json").write_text(run.model_dump_json(indent=2))
```

**Step 4: Write `__init__.py`**

```python
from .store import RunStore
from .types import Run, RunConfig, RunStatus

__all__ = ["RunStore", "Run", "RunConfig", "RunStatus"]
```

**Step 5: Run tests**

Create `sidecar/tests/runs/__init__.py` (empty).
Run: `pytest sidecar/tests/runs -v`
Expected: PASS

**Step 6: Commit**

```bash
git add sidecar/llm_chain_sidecar/runs/ sidecar/tests/runs/
git commit -m "feat(runs): JSON-on-disk run store with status updates"
```

---

## Task 9: Trainer interface + HF/CUDA backend (LoRA)

**Files:**
- Create: `sidecar/llm_chain_sidecar/trainers/__init__.py`
- Create: `sidecar/llm_chain_sidecar/trainers/base.py`
- Create: `sidecar/llm_chain_sidecar/trainers/hf_cuda.py`
- Create: `sidecar/tests/trainers/test_base.py`
- Create: `sidecar/tests/trainers/test_hf_cuda.py`

**Step 1: Write failing test for the trainer base class**

```python
from llm_chain_sidecar.trainers.base import Trainer, TrainingEvent, EventType
from llm_chain_sidecar.runs.types import RunConfig


def test_trainer_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        Trainer(RunConfig(model_id="m", backend="cpu", technique="lora",
                          dataset_path="/tmp/x", epochs=1))  # type: ignore


def test_event_construction():
    e = TrainingEvent(type=EventType.STEP, step=1, total_steps=10,
                      loss=2.3, lr=2e-4)
    assert e.step == 1
    assert e.loss == 2.3
```

**Step 2: Write `sidecar/llm_chain_sidecar/trainers/base.py`**

```python
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from pydantic import BaseModel
from llm_chain_sidecar.runs.types import RunConfig


class EventType(str, Enum):
    START = "start"
    STEP = "step"
    EPOCH_END = "epoch_end"
    DONE = "done"
    ERROR = "error"


class TrainingEvent(BaseModel):
    type: EventType
    step: int = 0
    total_steps: int = 0
    epoch: int = 0
    loss: float | None = None
    lr: float | None = None
    message: str | None = None


class Trainer(ABC):
    def __init__(self, config: RunConfig, output_dir: str) -> None:
        self.config = config
        self.output_dir = output_dir

    @abstractmethod
    def train(self) -> Iterator[TrainingEvent]:
        """Yield TrainingEvent updates as training progresses."""
```

**Step 3: Write failing test for the CUDA trainer (mock-heavy — no actual training in CI)**

```python
from unittest.mock import patch, MagicMock
from llm_chain_sidecar.trainers.hf_cuda import HfCudaTrainer
from llm_chain_sidecar.trainers.base import EventType
from llm_chain_sidecar.runs.types import RunConfig


def test_hf_cuda_yields_start_then_steps_then_done(tmp_path):
    cfg = RunConfig(model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
                    backend="cuda", technique="lora",
                    dataset_path="ignored", epochs=1, batch_size=1)
    trainer = HfCudaTrainer(cfg, output_dir=str(tmp_path))

    fake_callback_events = [
        {"step": 1, "loss": 2.0, "lr": 2e-4, "total_steps": 2},
        {"step": 2, "loss": 1.8, "lr": 2e-4, "total_steps": 2},
    ]
    with patch.object(trainer, "_run_training_loop", return_value=iter(fake_callback_events)):
        events = list(trainer.train())

    assert events[0].type == EventType.START
    assert events[1].type == EventType.STEP and events[1].loss == 2.0
    assert events[2].type == EventType.STEP and events[2].loss == 1.8
    assert events[-1].type == EventType.DONE
```

**Step 4: Write `sidecar/llm_chain_sidecar/trainers/hf_cuda.py`**

```python
from collections.abc import Iterator
from pathlib import Path
from .base import Trainer, TrainingEvent, EventType


class HfCudaTrainer(Trainer):
    """LoRA fine-tuning via Hugging Face transformers + peft on CUDA.

    The actual HF Trainer doesn't natively yield events, so we use a
    TrainerCallback that pushes onto a queue and bridge that to a generator.
    For unit tests we patch _run_training_loop to inject fake events.
    """

    def train(self) -> Iterator[TrainingEvent]:
        yield TrainingEvent(type=EventType.START, message=f"Loading {self.config.model_id}")
        try:
            for raw in self._run_training_loop():
                yield TrainingEvent(
                    type=EventType.STEP,
                    step=raw["step"],
                    total_steps=raw["total_steps"],
                    loss=raw.get("loss"),
                    lr=raw.get("lr"),
                )
        except Exception as e:
            yield TrainingEvent(type=EventType.ERROR, message=str(e))
            return
        yield TrainingEvent(type=EventType.DONE, message=f"Saved to {self.output_dir}")

    def _run_training_loop(self) -> Iterator[dict]:
        """Real implementation. Patched out in tests.

        Wires HF Trainer + peft.LoraConfig + a callback that pushes events.
        Implemented in detail in Task 10 (the real-training integration test).
        """
        import queue
        from threading import Thread
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer as HFTrainer,
            TrainerCallback,
        )
        from peft import LoraConfig, get_peft_model
        from llm_chain_sidecar.datasets.loader import load_dataset as ds_load
        from llm_chain_sidecar.datasets.types import DatasetSource, DatasetFormat

        rows = ds_load(DatasetSource(format=DatasetFormat.JSONL_CHAT, path=self.config.dataset_path))
        tok = AutoTokenizer.from_pretrained(self.config.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def to_text(row):
            return {"text": "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])}

        ds = Dataset.from_list([to_text(r) for r in rows])
        ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=512, padding="max_length"))

        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, torch_dtype=torch.bfloat16).to("cuda")
        peft_cfg = LoraConfig(r=self.config.lora_rank, lora_alpha=self.config.lora_alpha,
                              target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_cfg)

        events: queue.Queue[dict | None] = queue.Queue()

        class Cb(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    events.put({"step": state.global_step, "total_steps": state.max_steps,
                                "loss": logs["loss"], "lr": logs.get("learning_rate")})
            def on_train_end(self, args, state, control, **kw):
                events.put(None)

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=1, save_strategy="epoch", report_to="none",
        )
        hf = HFTrainer(model=model, args=args, train_dataset=ds, callbacks=[Cb()])

        Thread(target=hf.train, daemon=True).start()
        while True:
            ev = events.get()
            if ev is None:
                break
            yield ev
```

**Step 5: Run tests**

Create `sidecar/tests/trainers/__init__.py` (empty).
Run: `pytest sidecar/tests/trainers/test_base.py sidecar/tests/trainers/test_hf_cuda.py -v`
Expected: PASS (the real loop isn't invoked because we patch `_run_training_loop`)

**Step 6: Add ML deps**

Add to `pyproject.toml` `dependencies`:
```toml
"transformers>=4.40",
"trl>=0.8",
"peft>=0.10",
"accelerate>=0.30",
```

Add a new optional extra:
```toml
[project.optional-dependencies]
cuda = ["bitsandbytes>=0.43"]
```

Reinstall: `pip install -e ./sidecar[dev]` (CUDA extra installed only on supported boxes).

**Step 7: Commit**

```bash
git add sidecar/llm_chain_sidecar/trainers/ sidecar/tests/trainers/ sidecar/pyproject.toml
git commit -m "feat(trainers): HF + CUDA LoRA trainer with event-stream interface"
```

---

## Task 10: Trainer — Apple Silicon MLX backend

**Files:**
- Create: `sidecar/llm_chain_sidecar/trainers/mlx.py`
- Create: `sidecar/tests/trainers/test_mlx.py`

**Step 1: Write failing test (MLX trainer, mocked subprocess)**

```python
import sys
from unittest.mock import patch, MagicMock
import pytest
from llm_chain_sidecar.trainers.mlx import MlxTrainer
from llm_chain_sidecar.trainers.base import EventType
from llm_chain_sidecar.runs.types import RunConfig


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_yields_start_step_done(tmp_path):
    cfg = RunConfig(model_id="mlx-community/Qwen3-0.6B-4bit",
                    backend="mlx", technique="qlora",
                    dataset_path="ignored", epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(tmp_path))

    fake_lines = iter([
        b"Iter 10: Train loss 2.34, Learning Rate 1.0e-4\n",
        b"Iter 20: Train loss 1.98, Learning Rate 1.0e-4\n",
        b"",  # EOF
    ])
    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = fake_lines
    fake_proc.wait.return_value = 0
    with patch("llm_chain_sidecar.trainers.mlx.subprocess.Popen", return_value=fake_proc):
        events = list(trainer.train())

    assert events[0].type == EventType.START
    losses = [e.loss for e in events if e.type == EventType.STEP]
    assert losses == [2.34, 1.98]
    assert events[-1].type == EventType.DONE
```

**Step 2: Write `sidecar/llm_chain_sidecar/trainers/mlx.py`**

```python
import re
import subprocess
import sys
from collections.abc import Iterator
from .base import Trainer, TrainingEvent, EventType

# mlx_lm log format: "Iter N: Train loss V, Learning Rate L"
_LINE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\de.+\-]+)")


class MlxTrainer(Trainer):
    """LoRA/QLoRA fine-tuning via mlx_lm.lora subprocess on Apple Silicon."""

    def train(self) -> Iterator[TrainingEvent]:
        yield TrainingEvent(type=EventType.START, message=f"Spawning mlx_lm.lora on {self.config.model_id}")
        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", self.config.model_id,
            "--train", "--data", self.config.dataset_path,
            "--adapter-path", self.output_dir,
            "--iters", str(self.config.epochs * 100),
            "--batch-size", str(self.config.batch_size),
            "--learning-rate", str(self.config.learning_rate),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                m = _LINE.search(line.decode("utf-8", errors="replace"))
                if m:
                    yield TrainingEvent(
                        type=EventType.STEP,
                        step=int(m.group(1)), total_steps=self.config.epochs * 100,
                        loss=float(m.group(2)), lr=float(m.group(3)),
                    )
            rc = proc.wait()
            if rc != 0:
                yield TrainingEvent(type=EventType.ERROR, message=f"mlx_lm exited {rc}")
                return
        except Exception as e:
            yield TrainingEvent(type=EventType.ERROR, message=str(e))
            return
        yield TrainingEvent(type=EventType.DONE, message=f"Adapter saved to {self.output_dir}")
```

**Step 3: Add MLX as a Mac-only optional extra**

In `pyproject.toml`:
```toml
[project.optional-dependencies]
mlx = ["mlx>=0.15", "mlx-lm>=0.18"]
```

(MLX wheels only resolve on macOS arm64; on other OSes installation of the extra simply fails — handled by per-platform install in CI.)

**Step 4: Update `trainers/__init__.py`**

```python
import sys
from .base import Trainer, TrainingEvent, EventType
from .hf_cuda import HfCudaTrainer

__all__ = ["Trainer", "TrainingEvent", "EventType", "HfCudaTrainer", "make_trainer"]

if sys.platform == "darwin":
    from .mlx import MlxTrainer  # noqa: F401
    __all__.append("MlxTrainer")


def make_trainer(backend: str, *args, **kwargs) -> Trainer:
    if backend == "cuda":
        return HfCudaTrainer(*args, **kwargs)
    if backend == "mlx":
        if sys.platform != "darwin":
            raise RuntimeError("MLX backend requires macOS")
        return MlxTrainer(*args, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")
```

**Step 5: Run tests**

Run: `pytest sidecar/tests/trainers -v`
Expected: PASS on macOS; MLX test SKIPPED on Linux/Windows.

**Step 6: Commit**

```bash
git add sidecar/llm_chain_sidecar/trainers/ sidecar/tests/trainers/ sidecar/pyproject.toml
git commit -m "feat(trainers): MLX backend for Apple Silicon (subprocess wrapper)"
```

---

## Task 11: Run executor — connect Run + Trainer + persistence

**Files:**
- Create: `sidecar/llm_chain_sidecar/runs/executor.py`
- Create: `sidecar/tests/runs/test_executor.py`

**Step 1: Write failing test**

```python
from pathlib import Path
from unittest.mock import patch
from llm_chain_sidecar.runs.executor import RunExecutor
from llm_chain_sidecar.runs.store import RunStore
from llm_chain_sidecar.runs.types import RunConfig, RunStatus
from llm_chain_sidecar.trainers.base import TrainingEvent, EventType


class FakeTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
    def train(self):
        yield TrainingEvent(type=EventType.START)
        yield TrainingEvent(type=EventType.STEP, step=1, total_steps=2, loss=2.0)
        yield TrainingEvent(type=EventType.STEP, step=2, total_steps=2, loss=1.8)
        yield TrainingEvent(type=EventType.DONE)


def test_executor_runs_trainer_and_marks_succeeded(tmp_path: Path):
    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    executor = RunExecutor(store)
    with patch("llm_chain_sidecar.runs.executor.make_trainer", return_value=FakeTrainer(cfg, run.output_dir)):
        events = list(executor.execute(run.id))
    assert events[-1].type == EventType.DONE
    assert store.get(run.id).status == RunStatus.SUCCEEDED


def test_executor_marks_failed_on_error_event(tmp_path: Path):
    class FailingTrainer:
        def __init__(self, config, output_dir): pass
        def train(self):
            yield TrainingEvent(type=EventType.START)
            yield TrainingEvent(type=EventType.ERROR, message="boom")

    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    executor = RunExecutor(store)
    with patch("llm_chain_sidecar.runs.executor.make_trainer", return_value=FailingTrainer(cfg, "")):
        list(executor.execute(run.id))
    saved = store.get(run.id)
    assert saved.status == RunStatus.FAILED
    assert "boom" in (saved.error or "")
```

**Step 2: Write `sidecar/llm_chain_sidecar/runs/executor.py`**

```python
from collections.abc import Iterator
from llm_chain_sidecar.trainers import make_trainer
from llm_chain_sidecar.trainers.base import TrainingEvent, EventType
from .store import RunStore
from .types import RunStatus


class RunExecutor:
    def __init__(self, store: RunStore) -> None:
        self.store = store

    def execute(self, run_id: str) -> Iterator[TrainingEvent]:
        run = self.store.get(run_id)
        self.store.update_status(run_id, RunStatus.RUNNING)
        trainer = make_trainer(run.config.backend, run.config, run.output_dir)
        had_error = False
        try:
            for ev in trainer.train():
                if ev.type == EventType.ERROR:
                    had_error = True
                    self.store.update_status(run_id, RunStatus.FAILED, error=ev.message)
                yield ev
        except Exception as e:
            had_error = True
            self.store.update_status(run_id, RunStatus.FAILED, error=str(e))
            yield TrainingEvent(type=EventType.ERROR, message=str(e))
            return
        if not had_error:
            self.store.update_status(run_id, RunStatus.SUCCEEDED)
```

**Step 3: Run tests**

Run: `pytest sidecar/tests/runs/test_executor.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add sidecar/llm_chain_sidecar/runs/executor.py sidecar/tests/runs/test_executor.py
git commit -m "feat(runs): executor that bridges trainer events to store status"
```

---

## Task 12: FastAPI routes — hardware, models, datasets, runs (sync)

**Files:**
- Create: `sidecar/llm_chain_sidecar/api/__init__.py`
- Create: `sidecar/llm_chain_sidecar/api/routes.py`
- Modify: `sidecar/llm_chain_sidecar/main.py`
- Create: `sidecar/tests/api/test_routes.py`

**Step 1: Write failing tests**

```python
from fastapi.testclient import TestClient
from llm_chain_sidecar.main import app

client = TestClient(app)


def test_get_hardware():
    r = client.get("/api/hardware")
    assert r.status_code == 200
    j = r.json()
    assert "os" in j and "devices" in j


def test_get_models_default_excludes_restricted():
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    licenses = {m["license"] for m in models}
    assert licenses.issubset({"Apache-2.0", "MIT"})


def test_get_models_filtered_by_max_params():
    r = client.get("/api/models?max_params=500000000")
    assert all(m["params"] <= 500_000_000 for m in r.json()["models"])


def test_create_run_returns_id_and_lists():
    body = {
        "model_id": "Qwen/Qwen3-0.6B",
        "backend": "cuda",
        "technique": "lora",
        "dataset_path": "/tmp/x.jsonl",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200
    run_id = r.json()["id"]
    listing = client.get("/api/runs").json()["runs"]
    assert any(rn["id"] == run_id for rn in listing)
```

**Step 2: Write `sidecar/llm_chain_sidecar/api/routes.py`**

```python
from pathlib import Path
from fastapi import APIRouter, Query
from llm_chain_sidecar.hardware import probe_hardware
from llm_chain_sidecar.hardware.capabilities import capabilities_for_vram
from llm_chain_sidecar.models import ModelRegistry
from llm_chain_sidecar.runs.store import RunStore
from llm_chain_sidecar.runs.types import RunConfig

router = APIRouter(prefix="/api")
_store = RunStore(root=Path.home() / ".llm-chain" / "runs")
_registry = ModelRegistry.load_default()


@router.get("/hardware")
def get_hardware() -> dict:
    report = probe_hardware()
    devices = [d.model_dump() for d in report.devices]
    for d in devices:
        cap = capabilities_for_vram(d["vram_gb"], d["is_unified_memory"])
        d["capabilities"] = {
            "qlora_max_params": cap.qlora_max_params,
            "lora_max_params": cap.lora_max_params,
            "full_ft_max_params": cap.full_ft_max_params,
            "notes": cap.notes,
        }
    out = report.model_dump()
    out["devices"] = devices
    return out


@router.get("/models")
def get_models(max_params: int | None = Query(default=None)) -> dict:
    entries = _registry.entries
    if max_params is not None:
        entries = [e for e in entries if e.params <= max_params]
    return {"models": [e.model_dump() for e in entries]}


@router.post("/runs")
def create_run(cfg: RunConfig) -> dict:
    run = _store.create(cfg)
    return {"id": run.id, "status": run.status.value}


@router.get("/runs")
def list_runs() -> dict:
    return {"runs": [r.model_dump(mode="json") for r in _store.list()]}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _store.get(run_id).model_dump(mode="json")
```

**Step 3: Wire router in `main.py`**

```python
from fastapi import FastAPI
import uvicorn
from . import __version__
from .api.routes import router as api_router

app = FastAPI(title="LLM-Chain Sidecar", version=__version__)
app.include_router(api_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


def run() -> None:
    uvicorn.run(app, host="127.0.0.1", port=0)
```

**Step 4: Run tests**

Create `sidecar/tests/api/__init__.py` (empty).
Run: `pytest sidecar/tests/api -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sidecar/llm_chain_sidecar/api/ sidecar/llm_chain_sidecar/main.py sidecar/tests/api/
git commit -m "feat(api): /hardware, /models, /runs endpoints"
```

---

## Task 13: FastAPI route — SSE for live training events

**Files:**
- Modify: `sidecar/llm_chain_sidecar/api/routes.py`
- Create: `sidecar/tests/api/test_runs_stream.py`

**Step 1: Write failing test**

```python
from unittest.mock import patch
from fastapi.testclient import TestClient
from llm_chain_sidecar.main import app
from llm_chain_sidecar.trainers.base import TrainingEvent, EventType


def test_stream_run_emits_sse_events():
    fake_events = [
        TrainingEvent(type=EventType.START),
        TrainingEvent(type=EventType.STEP, step=1, total_steps=1, loss=1.0),
        TrainingEvent(type=EventType.DONE),
    ]
    client = TestClient(app)
    body = {"model_id": "m", "backend": "cuda", "technique": "lora",
            "dataset_path": "/tmp/x", "epochs": 1}
    run_id = client.post("/api/runs", json=body).json()["id"]
    with patch("llm_chain_sidecar.api.routes._executor.execute", return_value=iter(fake_events)):
        with client.stream("GET", f"/api/runs/{run_id}/stream") as r:
            body_text = "".join(r.iter_text())
    assert "event: start" in body_text
    assert "event: step" in body_text
    assert "event: done" in body_text
    assert '"loss":1.0' in body_text
```

**Step 2: Add to `sidecar/llm_chain_sidecar/api/routes.py`**

Add at the top:
```python
from fastapi.responses import StreamingResponse
from llm_chain_sidecar.runs.executor import RunExecutor

_executor = RunExecutor(_store)
```

Add the route:
```python
@router.get("/runs/{run_id}/stream")
def stream_run(run_id: str) -> StreamingResponse:
    def gen():
        for ev in _executor.execute(run_id):
            payload = ev.model_dump_json()
            yield f"event: {ev.type.value}\ndata: {payload}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
```

**Step 3: Run test**

Run: `pytest sidecar/tests/api/test_runs_stream.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add sidecar/llm_chain_sidecar/api/routes.py sidecar/tests/api/test_runs_stream.py
git commit -m "feat(api): SSE stream for live training events"
```

---

## Task 14: End-to-end real-training smoke test (gated)

**Files:**
- Create: `sidecar/tests/e2e/test_real_training.py`
- Create: `sidecar/tests/e2e/fixtures/tiny.jsonl`
- Modify: `pyproject.toml` (add a `slow` pytest marker)

**Step 1: Add the marker config**

In `sidecar/pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = ["slow: real training tests gated by hardware"]
```

**Step 2: Create tiny fixture dataset `sidecar/tests/e2e/fixtures/tiny.jsonl`**

```json
{"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]}
{"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}]}
```

**Step 3: Write the gated e2e test**

```python
import sys
from pathlib import Path
import pytest
import torch

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers import make_trainer
from llm_chain_sidecar.trainers.base import EventType


FIXTURE = Path(__file__).parent / "fixtures" / "tiny.jsonl"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_lora_step_on_cuda(tmp_path):
    cfg = RunConfig(
        model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        backend="cuda", technique="lora",
        dataset_path=str(FIXTURE), epochs=1, batch_size=1, lora_rank=4, lora_alpha=8,
    )
    trainer = make_trainer("cuda", cfg, str(tmp_path))
    events = list(trainer.train())
    types = [e.type for e in events]
    assert EventType.START in types
    assert EventType.STEP in types
    assert EventType.DONE in types
    losses = [e.loss for e in events if e.type == EventType.STEP and e.loss is not None]
    assert len(losses) >= 1


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_real_lora_step_on_mlx(tmp_path):
    cfg = RunConfig(
        model_id="mlx-community/Qwen3-0.6B-4bit",
        backend="mlx", technique="qlora",
        dataset_path=str(FIXTURE), epochs=1, batch_size=1,
    )
    trainer = make_trainer("mlx", cfg, str(tmp_path))
    events = list(trainer.train())
    types = [e.type for e in events]
    assert EventType.START in types
    assert EventType.DONE in types
```

**Step 4: Run only on a GPU host**

Run: `pytest sidecar/tests/e2e -v -m slow`
Expected on a 12GB+ NVIDIA box or 16GB+ Mac: PASS within ~3 minutes.
On CI without GPUs: SKIPPED.

**Step 5: Commit**

```bash
git add sidecar/tests/e2e/ sidecar/pyproject.toml
git commit -m "test(e2e): real LoRA training smoke tests gated by hardware"
```

---

## Task 15: Tauri shell scaffold

**Files:**
- Create: `apps/desktop/` via `npm create tauri-app`
- Modify: `apps/desktop/src-tauri/tauri.conf.json`
- Modify: `apps/desktop/package.json`

**Step 1: Scaffold**

```bash
cd apps && npm create tauri-app@latest desktop -- --template react-ts --yes
cd desktop && npm install
```

**Step 2: Verify it runs**

Run: `npm run tauri dev`
Expected: Tauri window opens with the React starter. Close it.

**Step 3: Configure Tauri identifier and product name**

Edit `apps/desktop/src-tauri/tauri.conf.json`:

```json
{
  "productName": "LLM-Chain",
  "identifier": "dev.llm-chain.app",
  "build": {
    "frontendDist": "../dist",
    "devUrl": "http://localhost:1420",
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build"
  },
  "app": {
    "windows": [
      {
        "title": "LLM-Chain — Train Your Own LLM",
        "width": 1280,
        "height": 800,
        "minWidth": 1024,
        "minHeight": 700
      }
    ]
  }
}
```

**Step 4: Commit**

```bash
git add apps/desktop/
git commit -m "feat(desktop): scaffold Tauri 2 + React + TypeScript shell"
```

---

## Task 16: Tauri sidecar wiring — bundle the Python sidecar binary

**Files:**
- Create: `apps/desktop/src-tauri/binaries/` (placeholder)
- Create: `scripts/build-sidecar.sh` (Mac/Linux), `scripts/build-sidecar.ps1` (Windows)
- Modify: `apps/desktop/src-tauri/tauri.conf.json`
- Modify: `apps/desktop/src-tauri/src/lib.rs`

**Step 1: Write `scripts/build-sidecar.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Build the sidecar into a standalone binary using PyInstaller.
# Output: apps/desktop/src-tauri/binaries/llm-chain-sidecar-<triple>
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRIPLE="$(rustc -vV | grep host | awk '{print $2}')"
OUT="$ROOT/apps/desktop/src-tauri/binaries"
mkdir -p "$OUT"

cd "$ROOT/sidecar"
pip install pyinstaller
pyinstaller --onefile --name "llm-chain-sidecar-${TRIPLE}" \
    --distpath "$OUT" --workpath /tmp/llm-chain-build --specpath /tmp/llm-chain-build \
    -p . llm_chain_sidecar/main.py
echo "Built: $OUT/llm-chain-sidecar-${TRIPLE}"
```

PowerShell variant `scripts/build-sidecar.ps1`:

```powershell
$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/.."
$triple = (rustc -vV | Select-String "host:").ToString().Split()[-1]
$out = "$root/apps/desktop/src-tauri/binaries"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Push-Location "$root/sidecar"
pip install pyinstaller
pyinstaller --onefile --name "llm-chain-sidecar-$triple.exe" `
    --distpath $out --workpath $env:TEMP/llm-chain-build --specpath $env:TEMP/llm-chain-build `
    -p . llm_chain_sidecar/main.py
Pop-Location
```

Make executable: `chmod +x scripts/build-sidecar.sh`.

**Step 2: Register the sidecar in Tauri config**

Add to `tauri.conf.json` under `bundle`:

```json
"bundle": {
  "active": true,
  "targets": "all",
  "externalBin": ["binaries/llm-chain-sidecar"]
}
```

Tauri auto-appends the platform triple (`-aarch64-apple-darwin`, `-x86_64-pc-windows-msvc`, etc.).

**Step 3: Spawn the sidecar from Rust**

Add `tauri-plugin-shell = "2"` to `apps/desktop/src-tauri/Cargo.toml` `[dependencies]`.

Edit `apps/desktop/src-tauri/src/lib.rs`:

```rust
use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;
use std::sync::Mutex;

#[derive(Default)]
struct SidecarState {
    port: Mutex<Option<u16>>,
}

#[tauri::command]
fn sidecar_port(state: tauri::State<SidecarState>) -> Option<u16> {
    *state.port.lock().unwrap()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState::default())
        .setup(|app| {
            let handle = app.handle().clone();
            let sidecar = handle.shell().sidecar("llm-chain-sidecar")?.spawn()?;
            let (mut rx, _child) = sidecar;
            let state: tauri::State<SidecarState> = handle.state();
            let port_state = state.port.clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    if let CommandEvent::Stdout(line) = event {
                        let s = String::from_utf8_lossy(&line);
                        // Sidecar prints "LLM_CHAIN_SIDECAR_PORT=<n>" on startup
                        if let Some(rest) = s.strip_prefix("LLM_CHAIN_SIDECAR_PORT=") {
                            if let Ok(p) = rest.trim().parse::<u16>() {
                                *port_state.lock().unwrap() = Some(p);
                            }
                        }
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![sidecar_port])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**Step 4: Update sidecar `main.py` to print its port on startup**

Edit `sidecar/llm_chain_sidecar/main.py` `run()`:

```python
def run() -> None:
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    print(f"LLM_CHAIN_SIDECAR_PORT={port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
```

**Step 5: Build sidecar + run end-to-end**

Run on Mac: `./scripts/build-sidecar.sh && cd apps/desktop && npm run tauri dev`
Expected: Tauri opens, devtools console shows the sidecar port via `invoke('sidecar_port')`.

**Step 6: Commit**

```bash
git add scripts/ apps/desktop/src-tauri/ sidecar/llm_chain_sidecar/main.py
git commit -m "feat(desktop): bundle Python sidecar via Tauri externalBin"
```

---

## Task 17: Frontend API client + state hook

**Files:**
- Create: `apps/desktop/src/api/client.ts`
- Create: `apps/desktop/src/api/hooks.ts`
- Create: `apps/desktop/src/api/__tests__/client.test.ts`
- Modify: `apps/desktop/package.json` (add `vitest`, `@tanstack/react-query`)

**Step 1: Install deps**

```bash
cd apps/desktop && npm install @tanstack/react-query
npm install -D vitest @vitest/ui jsdom
```

**Step 2: Write failing test for the client**

```typescript
import { describe, it, expect, vi } from "vitest";
import { ApiClient } from "../client";

describe("ApiClient", () => {
  it("constructs URLs against the resolved sidecar port", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, json: async () => ({ os: "Darwin", devices: [] }),
    });
    const c = new ApiClient(8123, fetchMock as unknown as typeof fetch);
    const r = await c.getHardware();
    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:8123/api/hardware");
    expect(r.os).toBe("Darwin");
  });

  it("createRun POSTs the config", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, json: async () => ({ id: "abc", status: "pending" }),
    });
    const c = new ApiClient(8123, fetchMock as unknown as typeof fetch);
    const r = await c.createRun({
      model_id: "Qwen/Qwen3-0.6B", backend: "cuda", technique: "lora",
      dataset_path: "/tmp/x", epochs: 1,
    });
    expect(r.id).toBe("abc");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8123/api/runs",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
```

**Step 3: Write `apps/desktop/src/api/client.ts`**

```typescript
export interface HardwareDevice {
  backend: string;
  name: string;
  vram_gb: number;
  is_unified_memory: boolean;
  capabilities: {
    qlora_max_params: number;
    lora_max_params: number;
    full_ft_max_params: number;
    notes: string;
  };
}

export interface HardwareReport {
  os: string;
  cpu: { cores: number; name: string };
  system_ram_gb: number;
  devices: HardwareDevice[];
}

export interface ModelEntry {
  id: string;
  name: string;
  family: string;
  params: number;
  license: string;
  modalities: string[];
}

export interface RunConfig {
  model_id: string;
  backend: string;
  technique: "lora" | "qlora";
  dataset_path: string;
  dataset_format?: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  lora_rank?: number;
  lora_alpha?: number;
}

export class ApiClient {
  constructor(private port: number, private fetchImpl: typeof fetch = fetch) {}

  private base(path: string) { return `http://127.0.0.1:${this.port}${path}`; }

  async getHardware(): Promise<HardwareReport> {
    const r = await this.fetchImpl(this.base("/api/hardware"));
    return r.json();
  }
  async getModels(maxParams?: number): Promise<{ models: ModelEntry[] }> {
    const q = maxParams ? `?max_params=${maxParams}` : "";
    const r = await this.fetchImpl(this.base(`/api/models${q}`));
    return r.json();
  }
  async createRun(cfg: RunConfig): Promise<{ id: string; status: string }> {
    const r = await this.fetchImpl(this.base("/api/runs"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    return r.json();
  }
  streamRun(runId: string, onEvent: (ev: { type: string; payload: unknown }) => void): () => void {
    const es = new EventSource(this.base(`/api/runs/${runId}/stream`));
    ["start", "step", "epoch_end", "done", "error"].forEach((t) => {
      es.addEventListener(t, (e: MessageEvent) => onEvent({ type: t, payload: JSON.parse(e.data) }));
    });
    return () => es.close();
  }
}
```

**Step 4: Write `apps/desktop/src/api/hooks.ts`**

```typescript
import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";
import { ApiClient } from "./client";

export function useApiClient(): ApiClient | null {
  const [client, setClient] = useState<ApiClient | null>(null);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (let i = 0; i < 50 && !cancelled; i++) {
        const port = await invoke<number | null>("sidecar_port");
        if (port) { setClient(new ApiClient(port)); return; }
        await new Promise((r) => setTimeout(r, 100));
      }
    })();
    return () => { cancelled = true; };
  }, []);
  return client;
}
```

**Step 5: Add `vitest` config**

`apps/desktop/vitest.config.ts`:
```typescript
import { defineConfig } from "vitest/config";
export default defineConfig({ test: { environment: "jsdom" } });
```

Add to `package.json` scripts: `"test": "vitest run"`.

**Step 6: Run tests**

Run: `cd apps/desktop && npm test`
Expected: PASS

**Step 7: Commit**

```bash
git add apps/desktop/src/api/ apps/desktop/vitest.config.ts apps/desktop/package.json apps/desktop/package-lock.json
git commit -m "feat(desktop): API client + sidecar-port hook"
```

---

## Task 18: UI — install Tailwind, build the four screens

**Files:**
- Modify: `apps/desktop/package.json`, Tailwind config files
- Create: `apps/desktop/src/screens/Dashboard.tsx`
- Create: `apps/desktop/src/screens/ModelPicker.tsx`
- Create: `apps/desktop/src/screens/DatasetPicker.tsx`
- Create: `apps/desktop/src/screens/Train.tsx`
- Create: `apps/desktop/src/screens/Runs.tsx`
- Modify: `apps/desktop/src/App.tsx`, `apps/desktop/src/main.tsx`

**Step 1: Install Tailwind v4 + Recharts + react-router**

```bash
cd apps/desktop
npm install -D tailwindcss @tailwindcss/vite
npm install recharts react-router-dom
```

Edit `apps/desktop/vite.config.ts` to add Tailwind plugin.

Edit `apps/desktop/src/index.css`:
```css
@import "tailwindcss";
```

**Step 2: Write `Dashboard.tsx` — hardware report**

```tsx
import { useApiClient } from "../api/hooks";
import { useEffect, useState } from "react";
import type { HardwareReport } from "../api/client";

export function Dashboard() {
  const api = useApiClient();
  const [hw, setHw] = useState<HardwareReport | null>(null);
  useEffect(() => { if (api) api.getHardware().then(setHw); }, [api]);
  if (!api || !hw) return <div className="p-6">Probing hardware…</div>;
  return (
    <div className="p-6 space-y-4">
      <h1 className="text-2xl font-semibold">Your Machine</h1>
      <div className="text-sm text-zinc-500">{hw.os} • {hw.cpu.cores} cores • {hw.system_ram_gb} GB RAM</div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {hw.devices.map((d, i) => (
          <div key={i} className="rounded-lg border p-4">
            <div className="font-medium">{d.name}</div>
            <div className="text-xs uppercase text-zinc-500">{d.backend}</div>
            <div className="mt-2 text-sm">
              {d.vram_gb > 0 ? `${d.vram_gb} GB ${d.is_unified_memory ? "unified" : "VRAM"}` : "CPU only"}
            </div>
            <div className="mt-2 text-xs text-zinc-600">
              QLoRA up to {(d.capabilities.qlora_max_params / 1e9).toFixed(1)}B<br/>
              LoRA up to {(d.capabilities.lora_max_params / 1e9).toFixed(1)}B<br/>
              Full FT up to {(d.capabilities.full_ft_max_params / 1e6).toFixed(0)}M
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

**Step 3: Write `ModelPicker.tsx`** — list models, gated by selected device's QLoRA cap (full screen code in commit; pattern: select a device → fetch `/api/models?max_params=<cap>` → render cards with license badge → store selection in zustand or React context).

**Step 4: Write `DatasetPicker.tsx`** — file picker (Tauri `dialog` plugin) for JSONL/CSV, text input for HF id, format dropdown.

**Step 5: Write `Train.tsx`** — show selected model + dataset + technique radio + hyperparams form + Start button. On start: `api.createRun(cfg)` then redirect to `Runs` for the new id.

**Step 6: Write `Runs.tsx`** — list all runs from `/api/runs`. Click → run detail with live SSE chart (Recharts line of loss vs step) + log tail. Subscribe via `api.streamRun(id, onEvent)`.

**Step 7: Wire routes in `App.tsx`** with `react-router-dom`. Sidebar nav.

**Step 8: Run dev server and click through**

```bash
npm run tauri dev
```
Expected: Sidecar reports hardware, model picker grays out anything bigger than the device cap, train flow navigates correctly.

**Step 9: Commit**

```bash
git add apps/desktop/
git commit -m "feat(ui): dashboard, model/dataset pickers, train and runs screens"
```

---

## Task 19: Bundling — produce signed Mac DMG and Windows MSI

**Files:**
- Modify: `.github/workflows/release.yml` (new)
- Modify: `apps/desktop/src-tauri/tauri.conf.json` (signing identifiers)

**Step 1: Add `.github/workflows/release.yml`**

```yaml
name: Release
on:
  push:
    tags: ["v*"]
jobs:
  build:
    strategy:
      matrix:
        include:
          - os: macos-14
            target: aarch64-apple-darwin
          - os: windows-latest
            target: x86_64-pc-windows-msvc
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - uses: dtolnay/rust-toolchain@stable
        with: { targets: ${{ matrix.target }} }
      - run: pip install -e ./sidecar
      - name: Build sidecar (macOS/Linux)
        if: runner.os != 'Windows'
        run: ./scripts/build-sidecar.sh
      - name: Build sidecar (Windows)
        if: runner.os == 'Windows'
        run: pwsh ./scripts/build-sidecar.ps1
      - run: cd apps/desktop && npm ci && npm run tauri build -- --target ${{ matrix.target }}
      - uses: actions/upload-artifact@v4
        with:
          name: llm-chain-${{ matrix.target }}
          path: |
            apps/desktop/src-tauri/target/${{ matrix.target }}/release/bundle/dmg/*
            apps/desktop/src-tauri/target/${{ matrix.target }}/release/bundle/msi/*
```

(Code signing requires secrets — `APPLE_CERTIFICATE`, `APPLE_ID`, `APPLE_TEAM_ID`, `WINDOWS_CERTIFICATE` — added later when the project gets a paid Apple Developer ID and a Windows code-signing cert.)

**Step 2: Tag & push to verify the workflow**

```bash
git tag v0.1.0-alpha
git push --tags
```
Expected: GitHub Actions produces `.dmg` and `.msi` artifacts (unsigned in this first pass).

**Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: release workflow building Mac DMG + Windows MSI"
```

---

## Task 20: Documentation — README, dev setup, supported hardware

**Files:**
- Modify: `README.md`
- Create: `docs/development.md`
- Create: `docs/supported-hardware.md`

**Step 1: Expand `README.md`**

- What it is, what it does, screenshots placeholder.
- Quick install (download from Releases).
- Supported hardware matrix.
- "Coming in v1.1 / v1.2".

**Step 2: Write `docs/development.md`**

- Prereqs (Python 3.11, Rust, Node 20).
- `pip install -e ./sidecar[dev]`, `cd apps/desktop && npm install`, `npm run tauri dev`.
- Running tests: `pytest sidecar`, `cd apps/desktop && npm test`.
- Building installers: `./scripts/build-sidecar.sh && cd apps/desktop && npm run tauri build`.

**Step 3: Write `docs/supported-hardware.md`**

Reproduce the VRAM-tier table from the design doc, with the v1.0 caveat ("CUDA + Apple Silicon only this release").

**Step 4: Commit**

```bash
git add README.md docs/development.md docs/supported-hardware.md
git commit -m "docs: developer setup, supported hardware, expanded readme"
```

---

## What v1.0 ships

After Task 20:
- A signed (eventually) Mac `.dmg` and Windows `.msi` bundling Tauri shell + Python sidecar
- Hardware probe + capability gating
- Curated Apache/MIT model picker
- JSONL / CSV / text-dir / HF Hub datasets
- LoRA/QLoRA fine-tune on NVIDIA CUDA via HF transformers + peft
- LoRA/QLoRA fine-tune on Apple Silicon via mlx-lm subprocess
- Live SSE training-event stream rendered as loss chart + log tail
- Local-only checkpoint output

## What v1.1 will add (separate plan)

- AMD ROCm backend (Linux + Windows for RX 7000/9000)
- Intel XPU backend (IPEX-LLM)
- CPU fallback for ≤100M models
- Unsloth integration on NVIDIA (faster + smaller VRAM)
- Restricted-license toggle (Llama, Gemma, DeepSeek base) with inline warning UI
- GGUF export (via llama.cpp converter)
- Hugging Face Hub push
- Opt-in anonymous telemetry

## What v1.2 will add (separate plan)

- Pretraining-from-scratch path (nanoGPT/nanochat) with appropriate hardware gating
- Multimodal training (Qwen2.5-VL, Idefics2/3, Phi-4-multimodal)
- Image-folder + image+caption parquet dataset loaders
- Vision-encoder branch in trainer dispatch
