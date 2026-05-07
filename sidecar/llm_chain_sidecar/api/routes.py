import json
import os
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from llm_chain_sidecar import exports
from llm_chain_sidecar.hardware import probe_hardware
from llm_chain_sidecar.hardware.capabilities import (
    capabilities_for_cpu,
    capabilities_for_vram,
)
from llm_chain_sidecar.models import ModelRegistry
from llm_chain_sidecar.runs.executor import RunExecutor
from llm_chain_sidecar.runs.store import RunStore
from llm_chain_sidecar.runs.types import RunConfig, RunStatus

router = APIRouter(prefix="/api")

_DEFAULT_RUNS_ROOT = Path.home() / ".llm-chain" / "runs"
_runs_root = Path(os.environ.get("LLM_CHAIN_RUNS_DIR", str(_DEFAULT_RUNS_ROOT)))
_store = RunStore(root=_runs_root)
_executor = RunExecutor(_store)
_registry = ModelRegistry.load_default()

_GGUF_STATE_FILE = "export-gguf.json"


def _gguf_state_path(run_id: str) -> Path:
    return _runs_root / run_id / _GGUF_STATE_FILE


def _read_gguf_state(run_id: str) -> dict | None:
    p = _gguf_state_path(run_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _write_gguf_state(run_id: str, state: dict) -> None:
    _gguf_state_path(run_id).write_text(json.dumps(state))


def _run_gguf_export(run_id: str, quant: str) -> None:
    """Background worker. Status transitions go through the state file so the
    GET endpoint can resolve progress without holding any in-memory handle."""
    try:
        _write_gguf_state(run_id, {"status": "running", "step": "merge", "quant": quant})
        merged = exports.merge_adapter(run_id, _runs_root)
        _write_gguf_state(run_id, {"status": "running", "step": "convert", "quant": quant})
        path = exports.convert_to_gguf(merged, quant=quant)
        _write_gguf_state(run_id, {"status": "done", "path": str(path), "quant": quant})
    except Exception as e:  # noqa: BLE001 — surface the failure back to the UI verbatim
        _write_gguf_state(run_id, {"status": "failed", "error": str(e), "quant": quant})


@router.get("/system/stats")
def get_system_stats() -> dict:
    """Live CPU / RAM / GPU snapshot for the top-bar indicator.

    Cheap to call (sub-100 ms). Polled by the UI every couple of seconds. We
    return absolute MB / percent so the UI doesn't need to know how much RAM
    the box has — the Dashboard already owns that.
    """
    import psutil

    vm = psutil.virtual_memory()
    out: dict = {
        "cpu_percent": psutil.cpu_percent(interval=None),  # non-blocking
        "ram": {
            "used_gb": round((vm.total - vm.available) / (1024**3), 2),
            "total_gb": round(vm.total / (1024**3), 2),
            "percent": vm.percent,
        },
        "gpu": None,
    }
    # GPU stats are best-effort. CUDA via torch; Apple unified via psutil's
    # process tree (we approximate with system RAM since the GPU shares it).
    try:
        import torch
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(i)
            out["gpu"] = {
                "name": torch.cuda.get_device_name(i),
                "vram_used_gb": round((total - free) / (1024**3), 2),
                "vram_total_gb": round(total / (1024**3), 2),
                "vram_percent": round((total - free) / total * 100, 1),
            }
    except Exception:
        pass  # CUDA not present or unhappy; UI just shows CPU + RAM
    return out


@router.get("/hardware")
def get_hardware() -> dict:
    report = probe_hardware()
    devices = [d.model_dump() for d in report.devices]
    for d in devices:
        if d["backend"] == "cpu":
            cap = capabilities_for_cpu()
        else:
            cap = capabilities_for_vram(d["vram_gb"], d["memory_kind"])
        d["capabilities"] = {
            "qlora_max_params": cap.qlora_max_params,
            "lora_max_params": cap.lora_max_params,
            "full_ft_max_params": cap.full_ft_max_params,
            "cpu_max_params": cap.cpu_max_params,
            "notes": cap.notes,
            "warning_codes": list(cap.warning_codes),
        }
    out = report.model_dump()
    out["devices"] = devices
    return out


@router.get("/models")
def get_models(
    max_params: int | None = Query(default=None),
    include_restricted: bool = Query(default=False),
    modalities: str | None = Query(default=None),
    chat_capable: bool = Query(default=False),
) -> dict:
    required = (
        [m.strip() for m in modalities.split(",") if m.strip()]
        if modalities
        else None
    )
    entries = _registry.entries(
        include_restricted=include_restricted,
        required_modalities=required,
        chat_capable_only=chat_capable,
    )
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


@router.get("/runs/{run_id}/stream")
def stream_run(run_id: str) -> StreamingResponse:
    def gen():
        for ev in _executor.execute(run_id):
            payload = ev.model_dump_json()
            yield f"event: {ev.type.value}\ndata: {payload}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    if not _executor.cancel(run_id):
        # Either the run never started streaming (no in-flight executor) or it
        # has already finished. 409 communicates "no active run to cancel".
        raise HTTPException(status_code=409, detail="run not active")
    return {"canceled": True}


@router.post("/runs/{run_id}/export/gguf")
def start_gguf_export(
    run_id: str, quant: str = Query(default="q4_k_m")
) -> JSONResponse:
    if quant not in exports.SUPPORTED_QUANTS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported quant '{quant}'; pick one of {sorted(exports.SUPPORTED_QUANTS)}",
        )
    try:
        run = _store.get(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="run not found") from e
    # Plan-spec: 404 covers both "run doesn't exist" and "run isn't ready".
    # 409 would be more semantic, but the UI just needs a definitive "no, not now".
    if run.status != RunStatus.SUCCEEDED:
        raise HTTPException(
            status_code=404,
            detail=f"run is {run.status.value}; gguf export requires a succeeded run",
        )
    state = _read_gguf_state(run_id)
    if state and state.get("status") == "running":
        # Don't start a duplicate. Echo current state so the UI can poll on.
        return JSONResponse(status_code=202, content=state)
    threading.Thread(
        target=_run_gguf_export, args=(run_id, quant), daemon=True
    ).start()
    return JSONResponse(
        status_code=202, content={"status": "running", "step": "merge", "quant": quant}
    )


@router.get("/runs/{run_id}/export/gguf")
def get_gguf_export(run_id: str) -> dict:
    try:
        _store.get(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="run not found") from e
    state = _read_gguf_state(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="no gguf export started for this run")
    return state


class _HubPushBody(BaseModel):
    repo_id: str
    private: bool = True
    folder: str = "adapter"


@router.get("/auth/hf")
def get_hf_auth_status() -> dict:
    return {"signed_in": exports.is_hf_signed_in()}


@router.post("/runs/{run_id}/export/hub")
def push_run_to_hub(run_id: str, body: _HubPushBody) -> dict:
    try:
        run = _store.get(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="run not found") from e
    if run.status != RunStatus.SUCCEEDED:
        raise HTTPException(
            status_code=404,
            detail=f"run is {run.status.value}; hub push requires a succeeded run",
        )
    try:
        url = exports.push_to_hub(
            run_id,
            body.repo_id,
            runs_root=_runs_root,
            private=body.private,
            folder=body.folder,
        )
    except exports.HubAuthError as e:
        # 401 reads cleaner in the UI than 500 — caller can prompt the user
        # to run `huggingface-cli login`.
        raise HTTPException(status_code=401, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"url": url, "repo_id": body.repo_id, "private": body.private}
