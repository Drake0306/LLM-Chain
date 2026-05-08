import json
import math
import os
import re
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from llm_chain_sidecar import exports
from llm_chain_sidecar.hardware import probe_hardware
from llm_chain_sidecar.hardware.capabilities import (
    capabilities_for_amd_vram,
    capabilities_for_cpu,
    capabilities_for_vram,
)
from llm_chain_sidecar.models import ModelRegistry
from llm_chain_sidecar.runs.executor import RunExecutor, read_events
from llm_chain_sidecar.runs.store import RunStore
from llm_chain_sidecar.runs.types import RunConfig, RunStatus
from llm_chain_sidecar.trainers.hf_rocm import is_experimental_armed

router = APIRouter(prefix="/api")

# Run IDs are uuid4().hex[:12] — 12 lowercase hex chars. Anything else has
# either been hand-crafted by an API caller or is path-traversal probing.
# Reject early so the storage layer never sees `../../etc/passwd` style
# paths and so 404 lookups on garbage IDs don't even hit the disk.
_RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=404, detail="run not found")


def _get_run_or_404(run_id: str):
    """Validate id shape, look up the run, raise 404 on either failure.

    Eight handlers had a copy-paste of this exact try/except. Keeping it
    in one place means the error envelope ('run not found') stays
    consistent and a future change (e.g. caching) only edits one site.
    """
    _validate_run_id(run_id)
    try:
        return _store.get(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="run not found") from e


def _get_succeeded_run_or_404(run_id: str, what: str):
    """Variant for the export endpoints, which require a SUCCEEDED run.

    ``what`` is a short noun ('gguf export', 'hub push') used in the
    error detail so the user can tell what they were attempting.
    """
    run = _get_run_or_404(run_id)
    if run.status != RunStatus.SUCCEEDED:
        # 404 is intentional rather than 409: the UI just needs a
        # definitive "no, not now"; 404 keeps the export endpoints'
        # error contract identical to the missing-run case.
        raise HTTPException(
            status_code=404,
            detail=f"run is {run.status.value}; {what} requires a succeeded run",
        )
    return run

_DEFAULT_RUNS_ROOT = Path.home() / ".llm-chain" / "runs"
_runs_root = Path(os.environ.get("LLM_CHAIN_RUNS_DIR", str(_DEFAULT_RUNS_ROOT)))
_store = RunStore(root=_runs_root)
_executor = RunExecutor(_store)
_registry = ModelRegistry.load_default()

_GGUF_STATE_FILE = "export-gguf.json"

# Tracks whether psutil.cpu_percent has been called at least once since the
# process started, so we can pay the 50 ms warmup cost on the first call only.
# psutil keeps its own internal baseline, so this dict only flips False→True.
_cpu_warm: dict[str, bool] = {"seen": False}


def _gguf_state_path(run_id: str) -> Path:
    return _runs_root / run_id / _GGUF_STATE_FILE


def _read_gguf_state(run_id: str) -> dict | None:
    p = _gguf_state_path(run_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        # Torn write from a concurrent _write_gguf_state. Caller will retry
        # on the next poll tick — better than surfacing a parse error to the
        # UI for what's an inherently transient race.
        return None


def _write_gguf_state(run_id: str, state: dict) -> None:
    """Atomic write: tmp-then-rename. The route's _read_gguf_state and the
    background worker's _set_state can race otherwise, and a concurrent read
    midway through write_text() reads truncated bytes that fail to parse.
    """
    target = _gguf_state_path(run_id)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, target)


def _run_gguf_export(run_id: str, quant: str) -> None:
    """Background worker. Status transitions go through the state file so the
    GET endpoint can resolve progress without holding any in-memory handle.

    On convert failure, we still surface the merged-model path so the user
    can use the standalone HF dir with mlx_lm.generate / transformers even
    without the llama.cpp tooling installed.
    """
    merged_path: str | None = None

    def _set_state(**fields) -> None:
        # Re-read so the latest_log we write doesn't clobber an earlier
        # merged_path (and vice versa).
        current = _read_gguf_state(run_id) or {}
        current.update(fields)
        _write_gguf_state(run_id, current)

    def _on_progress_merge(line: str) -> None:
        _set_state(latest_log=line)

    def _on_progress_convert(line: str) -> None:
        _set_state(latest_log=line)

    try:
        _set_state(status="running", step="merge", quant=quant, latest_log=None)
        merged = exports.merge_adapter(run_id, _runs_root, on_progress=_on_progress_merge)
        merged_path = str(merged)
        _set_state(
            status="running",
            step="convert",
            quant=quant,
            merged_path=merged_path,
            latest_log=None,
        )
        path = exports.convert_to_gguf(
            merged, quant=quant, on_progress=_on_progress_convert
        )
        _set_state(
            status="done",
            path=str(path),
            merged_path=merged_path,
            quant=quant,
            latest_log=None,
        )
    except Exception as e:  # noqa: BLE001 — surface the failure back to the UI verbatim
        msg = str(e)
        if merged_path and "convert_hf_to_gguf.py" in msg:
            msg += (
                f"\n\nThe merged model was saved at {merged_path} — you can "
                "load it directly with mlx_lm.generate or transformers without "
                "the llama.cpp tooling. Run scripts/llama-cpp-bootstrap.sh once "
                "to enable the GGUF step."
            )
        _set_state(
            status="failed",
            error=msg,
            quant=quant,
            merged_path=merged_path,
            latest_log=None,
        )


@router.get("/system/stats")
def get_system_stats() -> dict:
    """Live CPU / RAM / GPU snapshot for the top-bar indicator.

    Cheap to call (sub-100 ms). Polled by the UI every couple of seconds. We
    return absolute MB / percent so the UI doesn't need to know how much RAM
    the box has — the Dashboard already owns that.
    """
    import psutil

    # psutil.cpu_percent(interval=None) computes utilisation since the last
    # call. The very first call after process startup has no prior baseline
    # and returns 0.0, which made the UI's CPU bar flatline at 0% on the
    # initial render. interval=0.05 forces a 50 ms sample if we don't
    # already have a baseline so the first frame is meaningful.
    cpu_pct = psutil.cpu_percent(interval=None)
    if cpu_pct == 0.0 and not _cpu_warm["seen"]:
        cpu_pct = psutil.cpu_percent(interval=0.05)
    _cpu_warm["seen"] = True

    vm = psutil.virtual_memory()
    out: dict = {
        "cpu_percent": cpu_pct,
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
        elif d["backend"] == "rocm":
            cap = capabilities_for_amd_vram(d["vram_gb"])
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
    out["rocm_experimental_armed"] = is_experimental_armed()
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


_CHAT_FORMATS = {"jsonl_chat", "jsonl_chat_vision"}
_VISION_FORMAT = "jsonl_chat_vision"
_VLM_BACKENDS = {"cuda_vlm", "mlx_vlm"}
_LOCAL_FORMATS = {"jsonl_chat", "jsonl_chat_vision", "csv", "text_dir"}
_KNOWN_FORMATS = {"jsonl_chat", "jsonl_chat_vision", "csv", "text_dir", "hf_hub"}
# Path expectations per format. text_dir wants a directory; everything else
# wants a regular file. We enforce this so a user who picks a folder for a
# JSONL slot doesn't see a confusing IsADirectoryError mid-staging.
_FILE_FORMATS = {"jsonl_chat", "jsonl_chat_vision", "csv"}
_DIR_FORMATS = {"text_dir"}
_KNOWN_BACKENDS = {"cuda", "cuda_vlm", "rocm", "cpu", "mlx", "mlx_vlm"}
_KNOWN_TECHNIQUES = {"lora", "qlora"}


def _validate_run_config(cfg: RunConfig) -> None:
    """Reject combinations the trainers can't actually run, with a message
    that points at a fix instead of letting the user discover the failure
    via a 30-line mlx_lm/HF traceback.

    Validations are ordered cheapest-first (string checks before filesystem
    stat) so we fail fast on obviously broken configs.
    """
    # 0a. String enums. Pydantic accepts any string in these fields, so we
    # validate against the closed set the rest of the system understands.
    if cfg.dataset_format not in _KNOWN_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown dataset_format '{cfg.dataset_format}'. "
                f"Pick one of {sorted(_KNOWN_FORMATS)}."
            ),
        )
    if cfg.backend not in _KNOWN_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown backend '{cfg.backend}'. "
                f"Pick one of {sorted(_KNOWN_BACKENDS)}."
            ),
        )
    if cfg.technique not in _KNOWN_TECHNIQUES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown technique '{cfg.technique}'. "
                f"Pick one of {sorted(_KNOWN_TECHNIQUES)}."
            ),
        )

    # 0b. Numeric range checks. The HF Trainer / mlx_lm produce confusing
    # errors at zero/negative epochs or batch sizes, and a NaN learning rate
    # silently nukes the loss curve to NaN with no warning. Bound them.
    if cfg.epochs <= 0:
        raise HTTPException(status_code=400, detail="epochs must be >= 1.")
    if cfg.batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be >= 1.")
    if not math.isfinite(cfg.learning_rate) or cfg.learning_rate <= 0:
        raise HTTPException(
            status_code=400,
            detail="learning_rate must be a finite positive number.",
        )
    if cfg.lora_rank <= 0 or cfg.lora_rank > 512:
        raise HTTPException(
            status_code=400,
            detail="lora_rank must be between 1 and 512.",
        )
    if cfg.lora_alpha <= 0 or cfg.lora_alpha > 4096:
        raise HTTPException(
            status_code=400,
            detail="lora_alpha must be between 1 and 4096.",
        )

    # 1. Dataset path must exist for every format that consumes a local file.
    # HF Hub goes through the network so we don't validate it here.
    if cfg.dataset_format in _LOCAL_FORMATS:
        if not cfg.dataset_path:
            raise HTTPException(
                status_code=400,
                detail=f"dataset_path is required for {cfg.dataset_format}.",
            )
        ds_path = Path(cfg.dataset_path)
        if not ds_path.exists():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Dataset path does not exist: {cfg.dataset_path}. "
                    "Re-pick the file or folder on the Dataset page."
                ),
            )
        # Distinguish file-vs-dir so the user gets a precise hint instead of
        # an IsADirectoryError (or, worse, a 'No such file: x.jsonl/...').
        if cfg.dataset_format in _FILE_FORMATS and not ds_path.is_file():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Dataset path must be a file for {cfg.dataset_format}: "
                    f"{cfg.dataset_path}"
                ),
            )
        if cfg.dataset_format in _DIR_FORMATS and not ds_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Dataset path must be a directory for {cfg.dataset_format}: "
                    f"{cfg.dataset_path}"
                ),
            )
    elif cfg.dataset_format == "hf_hub":
        # The frontend stuffs the HF dataset id into dataset_path; reject
        # an empty value with the same shape of error.
        if not (cfg.dataset_path or "").strip():
            raise HTTPException(
                status_code=400,
                detail="HF Hub datasets need a dataset id (e.g. 'acme/dataset').",
            )

    # 2. CSV format requires text_column up front. Loader raises later
    # otherwise, but mid-training is too late and the error reads as a
    # mysterious ValueError.
    if cfg.dataset_format == "csv" and not (cfg.text_column or "").strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "CSV datasets need a text column. Set 'Text column' on the "
                "Dataset page to the column that holds the training text."
            ),
        )

    # 3. Backend/format/modality cross-check. The frontend resolves backend
    # automatically, but a hand-crafted POST or stale UI state could land
    # here with a mismatch — fail before we spawn a trainer that will only
    # crash later with a confusing tokenizer/model error.
    is_vision_dataset = cfg.dataset_format == _VISION_FORMAT
    is_vlm_backend = cfg.backend in _VLM_BACKENDS
    if is_vision_dataset and not is_vlm_backend:
        raise HTTPException(
            status_code=400,
            detail=(
                "Vision datasets need a VLM backend (cuda_vlm or mlx_vlm). "
                "Pick a vision-capable model on the Models page; the UI will "
                "route the run to the correct trainer."
            ),
        )
    if is_vlm_backend and not is_vision_dataset:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The '{cfg.backend}' backend only handles 'jsonl_chat_vision' "
                "datasets. Switch the dataset format or pick a non-VLM model."
            ),
        )

    # 4. Chat dataset on a registered base model with no chat template.
    if cfg.dataset_format in _CHAT_FORMATS:
        match = next(
            (e for e in _registry.entries(include_restricted=True) if e.id == cfg.model_id),
            None,
        )
        if match is not None and not match.chat_capable:
            chat_examples = [
                e.name
                for e in _registry.entries()
                if e.chat_capable and "image" not in e.modalities
            ][:3]
            suggestions = ", ".join(chat_examples) if chat_examples else "a chat-capable model"
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{match.name} is a base model with no chat template — "
                    f"the {cfg.dataset_format} dataset format won't work on it. "
                    f"Pick a chat-capable model (e.g. {suggestions}), or change "
                    "the dataset format to CSV / text-dir / HF Hub."
                ),
            )

    # 5. Vision dataset on a registered text-only model. The frontend strips
    # these but a direct API caller could still hit this combination.
    if is_vision_dataset:
        match = next(
            (e for e in _registry.entries(include_restricted=True) if e.id == cfg.model_id),
            None,
        )
        if match is not None and "image" not in match.modalities:
            vision_examples = [
                e.name for e in _registry.entries() if "image" in e.modalities
            ][:2]
            suggestions = ", ".join(vision_examples) if vision_examples else "a vision-language model"
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{match.name} is text-only — the chat-with-images dataset "
                    f"format needs a vision-language model (e.g. {suggestions})."
                ),
            )


@router.post("/runs")
def create_run(cfg: RunConfig) -> dict:
    _validate_run_config(cfg)
    run = _store.create(cfg)
    return {"id": run.id, "status": run.status.value}


@router.get("/runs")
def list_runs() -> dict:
    return {"runs": [r.model_dump(mode="json") for r in _store.list()]}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _get_run_or_404(run_id).model_dump(mode="json")


@router.get("/runs/{run_id}/events")
def get_run_events(run_id: str) -> dict:
    """Replay every event the trainer emitted for this run, in order. Used by
    the UI on RunDetail mount so loss curves, downloads, and log lines
    survive across navigation, app restarts, and SSE reconnects."""
    run = _get_run_or_404(run_id)
    return {"events": read_events(run.output_dir or _runs_root / run_id)}


@router.get("/runs/{run_id}/stream")
def stream_run(run_id: str) -> StreamingResponse:
    _validate_run_id(run_id)
    def gen():
        for ev in _executor.execute(run_id):
            payload = ev.model_dump_json()
            yield f"event: {ev.type.value}\ndata: {payload}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    _validate_run_id(run_id)
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
    _get_succeeded_run_or_404(run_id, "gguf export")
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
    _get_run_or_404(run_id)
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
    _get_succeeded_run_or_404(run_id, "hub push")
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
