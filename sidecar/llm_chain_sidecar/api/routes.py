import os
from pathlib import Path

from fastapi import APIRouter, Query

from llm_chain_sidecar.hardware import probe_hardware
from llm_chain_sidecar.hardware.capabilities import capabilities_for_vram
from llm_chain_sidecar.models import ModelRegistry
from llm_chain_sidecar.runs.store import RunStore
from llm_chain_sidecar.runs.types import RunConfig

router = APIRouter(prefix="/api")

_DEFAULT_RUNS_ROOT = Path.home() / ".llm-chain" / "runs"
_runs_root = Path(os.environ.get("LLM_CHAIN_RUNS_DIR", str(_DEFAULT_RUNS_ROOT)))
_store = RunStore(root=_runs_root)
_registry = ModelRegistry.load_default()


@router.get("/hardware")
def get_hardware() -> dict:
    report = probe_hardware()
    devices = [d.model_dump() for d in report.devices]
    for d in devices:
        cap = capabilities_for_vram(d["vram_gb"], d["memory_kind"])
        d["capabilities"] = {
            "qlora_max_params": cap.qlora_max_params,
            "lora_max_params": cap.lora_max_params,
            "full_ft_max_params": cap.full_ft_max_params,
            "notes": cap.notes,
            "warning_codes": list(cap.warning_codes),
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
