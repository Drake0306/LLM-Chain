import json
import math
import os
import re
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from llm_chain_sidecar import exports
from llm_chain_sidecar.hardware import probe_hardware
from llm_chain_sidecar.inference import GenerationConfig, generate_stream
from llm_chain_sidecar.inference.eval_suite import (
    DEFAULT_PROMPTS,
    EvalConfig,
    default_prompts_for_family,
    evaluate,
)
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
_HUB_STATE_FILE = "export-hub.json"

# Per-run "skip current eval prompt" flag. Set by POST .../eval/skip,
# observed by the eval orchestrator at the next token boundary, then
# cleared by the orchestrator before the next prompt starts. Single
# in-flight eval per run is the contract; concurrent evals would
# share a flag but that's not a supported workflow.
_eval_skip_events: dict[str, threading.Event] = {}


def _hf_hub_train_split_count(repo_id: str) -> int | None:
    """Cheap row count for a HF Hub dataset via its card metadata.

    Most HF dataset cards publish an auto-generated ``dataset_info``
    block listing each split with a ``num_examples`` field. Reading
    it requires only a single small HTTP fetch — no parquet
    streaming, no parquet reading. For datasets that don't ship that
    metadata (older or hand-curated ones) we return None and the UI
    falls back to "counted at training time".

    Robust to schema variation: HF's card metadata can come back as a
    plain dict or a typed object depending on the huggingface_hub
    version, and ``dataset_info`` can be a single dict or a list of
    configs. We probe both shapes and short-circuit on the first
    train split we recognise.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    try:
        info = HfApi().dataset_info(repo_id)
    except Exception:  # noqa: BLE001 — network / 404 / auth all map to "unknown"
        return None

    card = getattr(info, "card_data", None)
    if card is None:
        return None
    # card_data is sometimes a dict, sometimes a typed model.
    dsi = card.get("dataset_info") if isinstance(card, dict) else getattr(card, "dataset_info", None)
    if dsi is None:
        return None
    # dataset_info can be a single config dict or a list of configs
    # (multi-config datasets like c4). Pick the first config that
    # carries splits — good enough for the badge.
    if isinstance(dsi, list):
        dsi = next((d for d in dsi if d), None)
    if dsi is None:
        return None
    splits = dsi.get("splits") if isinstance(dsi, dict) else getattr(dsi, "splits", None)
    if not splits:
        return None
    # splits can be a list of {name, num_examples} entries or a dict
    # keyed by split name.
    if isinstance(splits, dict):
        splits = list(splits.values())
    for s in splits:
        name = s.get("name") if isinstance(s, dict) else getattr(s, "name", None)
        if name == "train":
            num = s.get("num_examples") if isinstance(s, dict) else getattr(s, "num_examples", None)
            try:
                return int(num) if num is not None else None
            except (TypeError, ValueError):
                return None
    return None

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
            cap = capabilities_for_vram(
                d["vram_gb"],
                d["memory_kind"],
                available_vram_gb=d.get("available_vram_gb"),
            )
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

    # 5a. Resume validation: if the user is continuing from an existing
    # run, that run must exist, be SUCCEEDED, share the same backend
    # family, and have an adapter file the trainer can read. Catching
    # this at the boundary avoids spawning the trainer just to fail
    # with a confusing FileNotFoundError mid-load.
    if cfg.resume_from is not None:
        if not _RUN_ID_RE.match(cfg.resume_from):
            raise HTTPException(
                status_code=400,
                detail="resume_from must reference a valid run id.",
            )
        try:
            parent = _store.get(cfg.resume_from)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=400,
                detail=f"resume_from run {cfg.resume_from} not found.",
            ) from e
        if parent.status != RunStatus.SUCCEEDED:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot resume from run {cfg.resume_from} — it's "
                    f"{parent.status.value}, not succeeded."
                ),
            )
        # Prevent backend mismatch — the adapter file format differs
        # between mlx (adapters.safetensors) and HF (adapter_model.
        # safetensors), and the LoRA shapes only line up when the same
        # base model + same rank/alpha are used.
        if parent.config.backend != cfg.backend:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Cannot resume across backends: parent run used "
                    f"{parent.config.backend!r}, this run uses "
                    f"{cfg.backend!r}."
                ),
            )
        if parent.config.model_id != cfg.model_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cannot resume on a different base model: parent used "
                    f"{parent.config.model_id!r}, this run uses "
                    f"{cfg.model_id!r}."
                ),
            )
        if (
            parent.config.lora_rank != cfg.lora_rank
            or parent.config.lora_alpha != cfg.lora_alpha
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cannot resume with different LoRA rank/alpha: parent "
                    f"used r={parent.config.lora_rank}/α={parent.config.lora_alpha}, "
                    f"this run uses r={cfg.lora_rank}/α={cfg.lora_alpha}."
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


class _DatasetPreviewBody(BaseModel):
    dataset_path: str
    dataset_format: str
    text_column: str | None = None
    # How many rows to return. Bounded so a hand-crafted POST can't
    # demand we materialise an entire 1M-row JSONL in memory.
    limit: int = 3


class _DatasetCountBody(BaseModel):
    """Body for POST /datasets/count.

    Mirrors the preview body but skips parsing: just returns the row
    count. Cheap enough that the Train page can call it on every
    dataset change without dragging the UI through a 1 GB JSONL parse.
    """
    dataset_path: str
    dataset_format: str
    text_column: str | None = None


@router.post("/datasets/count")
def count_dataset(body: _DatasetCountBody) -> dict:
    """Fast row count without loader-level parsing.

    The Train page's split badge needs to know whether the dataset is
    big enough to train on (≥ 2 rows) but doesn't need every row's
    contents materialised — running through ``load_dataset`` for that
    is wasteful on big files. We do the cheapest thing per format:
    JSONL → count non-empty lines; CSV → DictReader length; text_dir
    → glob count; HF Hub → skip (would force a download).
    """
    if body.dataset_format not in _KNOWN_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset_format '{body.dataset_format}'.",
        )
    if body.dataset_format == "hf_hub":
        # HF Hub datasets carry split-level row counts in their dataset
        # card metadata. This avoids the streaming-iterate-everything
        # cost of computing the count from scratch — for datasets that
        # publish the metadata (most well-maintained ones do), we
        # return an exact number cheaply. For ones that don't, fall
        # back to None and the UI shows "counted at training time".
        repo_id = (body.dataset_path or "").strip()
        if not repo_id:
            raise HTTPException(
                status_code=400,
                detail="HF Hub datasets need a repo id.",
            )
        return {
            "row_count": _hf_hub_train_split_count(repo_id),
            "format": body.dataset_format,
        }
    if not body.dataset_path or not Path(body.dataset_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset path does not exist: {body.dataset_path}",
        )
    p = Path(body.dataset_path)
    fmt = body.dataset_format
    try:
        if fmt in ("jsonl_chat", "jsonl_chat_vision"):
            # Count non-empty lines — cheaper than json.loads on each
            # row and matches what the loader will accept.
            count = 0
            with p.open("r", encoding="utf-8-sig") as f:
                for line in f:
                    if line.strip():
                        count += 1
            return {"row_count": count, "format": fmt}
        if fmt == "csv":
            import csv as _csv

            if not (body.text_column or "").strip():
                raise HTTPException(
                    status_code=400,
                    detail="CSV datasets need a text column.",
                )
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = _csv.reader(f)
                next(reader, None)  # skip header
                count = sum(1 for _ in reader)
            return {"row_count": count, "format": fmt}
        if fmt == "text_dir":
            if not p.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail="text_dir path must be a directory.",
                )
            count = sum(1 for q in p.rglob("*.txt") if q.is_file())
            return {"row_count": count, "format": fmt}
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"{p} is not valid UTF-8 (offset {e.start}).",
        ) from e
    raise HTTPException(
        status_code=400, detail=f"count not supported for {fmt}",
    )


class _DatasetBuildBody(BaseModel):
    """Body for POST /datasets/build (workshop write-out).

    Either ``raw_text`` or ``source_path`` must be set; the latter
    reads from disk so the UI can hand off large files via the Tauri
    file picker without serialising them through the JSON body.
    Cleaner toggles match :class:`workshop.CleaningOptions`. The
    output filename is derived from ``name`` if given, falling back
    to a timestamp slug so two un-named builds never overwrite.
    """

    raw_text: str | None = None
    source_path: str | None = None
    input_format: str  # csv | tsv | jsonl
    # Schema mapping. The route layer routes onto SchemaMapping; we
    # accept flat fields here so the JSON shape stays simple.
    target: str = "chat"  # chat | completion
    user_field: str | None = None
    assistant_field: str | None = None
    prompt_field: str | None = None
    completion_field: str | None = None
    passthrough_chat: bool = False
    # Cleaning toggles.
    drop_empty: bool = True
    dedupe: bool = True
    role_balance: bool = True
    max_chars: int | None = Field(default=None, ge=1, le=1_000_000)
    # Output naming. ``name`` becomes the JSONL stem under the datasets
    # dir; absent / blank → timestamped slug. ``output_path`` lets a
    # caller override the destination entirely (used by tests).
    name: str | None = None
    output_path: str | None = None


_BUILD_INPUT_FORMATS = {"csv", "tsv", "jsonl"}
_BUILD_TARGETS = {"chat", "completion"}


@router.post("/datasets/build")
def build_dataset(body: _DatasetBuildBody) -> dict:
    """Workshop write-out: parse → map → clean → JSONL.

    Returns the path to the new file plus per-stage drop counts so the
    UI can render a "kept N of M; dropped K duplicates, J empty" line
    after the build completes. The new file lives under
    ``~/.llm-chain/datasets/`` by default so the dataset picker can
    surface it without the user re-typing the path.
    """
    from llm_chain_sidecar.datasets import workshop as ws

    if body.input_format not in _BUILD_INPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown input_format '{body.input_format}'. "
                f"Pick one of {sorted(_BUILD_INPUT_FORMATS)}."
            ),
        )
    if body.target not in _BUILD_TARGETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown target '{body.target}'. Pick chat or completion.",
        )

    # 1. Source the raw text. Exactly one of raw_text / source_path must
    # be set — accepting both would let a caller silently override the
    # other and surprise users about which won.
    if (body.raw_text is None) == (body.source_path is None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of raw_text or source_path.",
        )
    if body.source_path is not None:
        src = Path(body.source_path)
        if not src.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"source_path does not exist: {body.source_path}",
            )
        try:
            text = src.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{src} is not valid UTF-8 (bad byte at offset {e.start}). "
                    "Re-save as UTF-8 and try again."
                ),
            ) from e
    else:
        text = body.raw_text or ""

    # 2. Parse + map. Both raise ValueError on bad shape — surface as 400.
    try:
        rows = ws.parse_text(text, body.input_format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not rows:
        raise HTTPException(
            status_code=400,
            detail=(
                "No rows found. Paste at least one data row (plus a "
                "header for CSV/TSV)."
            ),
        )

    schema = ws.SchemaMapping(
        target=body.target,  # type: ignore[arg-type]
        user_field=body.user_field,
        assistant_field=body.assistant_field,
        prompt_field=body.prompt_field,
        completion_field=body.completion_field,
        passthrough_chat=body.passthrough_chat,
    )
    try:
        mapped = ws.apply_schema(rows, schema)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # rows that fell off in apply_schema (missing user/assistant fields)
    # count as dropped_empty in the user-visible summary — the user
    # cares that data was lost, not which internal stage caught it.
    schema_drops = len(rows) - len(mapped)

    # 3. Clean. CleaningOptions has plain-old-data semantics; we just
    # forward the toggles from the body.
    cleaning = ws.CleaningOptions(
        drop_empty=body.drop_empty,
        dedupe=body.dedupe,
        role_balance=body.role_balance,
        max_chars=body.max_chars,
    )
    survivors, stats = ws.clean(mapped, cleaning)
    stats.input_rows = len(rows)
    stats.dropped_empty += schema_drops

    if not survivors:
        raise HTTPException(
            status_code=400,
            detail=(
                "All rows were dropped by the cleaners — relax the toggles "
                "or fix the source data and try again."
            ),
        )
    if len(survivors) < 2:
        # Same minimum the trainer enforces; surfacing here saves the
        # user a round-trip through Train + a confusing single-row error.
        raise HTTPException(
            status_code=400,
            detail=(
                "Workshop produced 1 row — training needs at least 2 so "
                "the train/validation split doesn't overlap. Add another "
                "row or relax cleaning toggles."
            ),
        )

    # 4. Resolve the output path. ``output_path`` (test override) wins,
    # then a sanitised name under the datasets dir, then a timestamp.
    if body.output_path:
        out = Path(body.output_path)
    else:
        from datetime import datetime, timezone

        if body.name:
            stem = ws.safe_filename(body.name)
        else:
            stem = "workshop-" + datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )
        out = ws.default_datasets_dir() / f"{stem}.jsonl"

    if out.suffix != ".jsonl":
        raise HTTPException(
            status_code=400,
            detail="Output path must end in .jsonl",
        )

    # Refuse to overwrite — the user might have built another dataset
    # with the same name. The UI will append a suffix and retry.
    if out.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"A file already exists at {out}. Pick a different name or "
                "delete the existing file."
            ),
        )

    bytes_written = ws.write_jsonl(survivors, out)
    return {
        "path": str(out),
        "bytes_written": bytes_written,
        "stats": stats.to_dict(),
    }


@router.post("/datasets/preview")
def preview_dataset(body: _DatasetPreviewBody) -> dict:
    """Return the first N rows of the user's dataset, parsed through the
    same loader the trainer uses. The Dataset picker shows this so users
    can confirm their data shape before clicking Train — catching format
    mistakes (single-row datasets, non-UTF-8, missing message keys,
    relative image paths to nonexistent files) right where the user
    can fix them.

    Bounded by ``limit`` (max 50). Errors come back as 400s with the
    loader's actual message so the user sees ''Row 3 missing role/content''
    instead of a stack trace.
    """
    from llm_chain_sidecar.datasets import (
        DatasetFormat,
        load_dataset as ds_load,
        make_source,
    )

    if body.dataset_format not in _KNOWN_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset_format '{body.dataset_format}'.",
        )
    if body.dataset_format in _LOCAL_FORMATS:
        if not body.dataset_path or not Path(body.dataset_path).exists():
            raise HTTPException(
                status_code=400,
                detail=f"Dataset path does not exist: {body.dataset_path}",
            )
    elif body.dataset_format == "hf_hub":
        if not (body.dataset_path or "").strip():
            raise HTTPException(
                status_code=400, detail="HF Hub datasets need a dataset id.",
            )
    if body.dataset_format == "csv" and not (body.text_column or "").strip():
        raise HTTPException(
            status_code=400, detail="CSV datasets need a text column.",
        )
    limit = max(1, min(int(body.limit or 3), 50))

    try:
        rows = ds_load(
            make_source(
                DatasetFormat(body.dataset_format),
                body.dataset_path,
                body.text_column,
            )
        )
    except (ValueError, FileNotFoundError) as e:
        # Loader errors carry actionable text already — pass through.
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "rows": rows[:limit],
        "row_count": len(rows),
        "shown": min(len(rows), limit),
    }


class _LrFinderBody(BaseModel):
    """POST body for the LR finder kickoff.

    The client supplies a base RunConfig (the one the user filled in
    on the Train page) plus an explicit list of LRs to probe. We
    create one short run per LR with ``max_steps`` clamped low and
    ``purpose="lr_finder"`` so the UI can group them.
    """
    config: RunConfig
    learning_rates: list[float] = Field(default_factory=lambda: [1e-4, 2e-4, 5e-4])
    steps_per_run: int = Field(default=10, ge=2, le=200)


class _SynthBody(BaseModel):
    """POST body for synthetic dataset generation.

    Either ``source_run_id`` (use an existing trained adapter / run)
    or both ``source_model_id`` + ``source_backend`` (synthesise from
    a fresh base model in the registry) must be set. ``count`` is
    bounded to keep a single SSE stream from running for hours.
    """

    source_run_id: str | None = None
    source_model_id: str | None = None
    source_backend: str | None = None
    topic: str = Field(min_length=1, max_length=2000)
    style: str = Field(default="", max_length=2000)
    count: int = Field(default=10, ge=1, le=100)
    max_tokens: int = Field(default=512, ge=64, le=4096)
    temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    seed_prompts: list[str] = Field(default_factory=list, max_length=20)


def _resolve_synth_run_dict(body: _SynthBody) -> dict:
    """Decide which run dict to feed the playground for this synth call.

    Validates the source-shape contract: exactly one of (run_id) or
    (model_id + backend) must be set. Returns either the stored run's
    dict or a synthesised base-only dict.
    """
    from llm_chain_sidecar.inference import synth as _synth

    has_run = body.source_run_id is not None
    has_base = body.source_model_id is not None and body.source_backend is not None
    if has_run == has_base:  # both or neither
        raise HTTPException(
            status_code=400,
            detail=(
                "Pick exactly one source: source_run_id (use a trained run) "
                "OR source_model_id + source_backend (use a base model)."
            ),
        )

    if has_run:
        run = _get_succeeded_run_or_404(body.source_run_id, "synth")
        if run.config.backend in _VLM_BACKENDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Synthetic data generation doesn't support vision-"
                    "language runs yet."
                ),
            )
        return run.model_dump(mode="json")

    if body.source_backend not in _KNOWN_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown source_backend '{body.source_backend}'. "
                f"Pick one of {sorted(_KNOWN_BACKENDS)}."
            ),
        )
    if body.source_backend in _VLM_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail="Synth from a vision-language base model isn't supported yet.",
        )
    # Validate the model is in the registry and chat-capable, otherwise
    # the chat-template builder will silently fall back to raw prompt
    # mode and the model won't follow the system instructions.
    match = next(
        (
            e
            for e in _registry.entries(include_restricted=True)
            if e.id == body.source_model_id
        ),
        None,
    )
    if match is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown source_model_id '{body.source_model_id}'. "
                "Pick a model from /api/models."
            ),
        )
    if not match.chat_capable:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{match.name} has no chat template — it can't follow the "
                "synth system prompt. Pick a chat-capable model."
            ),
        )
    return _synth.base_run_dict(body.source_model_id, body.source_backend)


@router.post("/datasets/synth")
def synth_dataset(body: _SynthBody) -> StreamingResponse:
    """Stream generated (user, assistant) conversation rows as SSE.

    Wire format mirrors /generate but with a ``row`` event carrying
    the parsed messages plus a parsed flag, terminated by ``done``
    with summary stats. The frontend collects rows in memory; saving
    is a separate POST to /api/datasets/build with passthrough_chat
    set, so the workshop's existing JSONL writer is reused.
    """
    from llm_chain_sidecar.inference import synth as _synth

    run_dict = _resolve_synth_run_dict(body)
    cfg = _synth.SynthConfig(
        topic=body.topic.strip(),
        style=body.style.strip(),
        count=body.count,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        seed_prompts=tuple(p.strip() for p in body.seed_prompts if p and p.strip()),
    )
    cancel_event = threading.Event()

    def gen():
        try:
            for frame in _synth.synthesize(
                run_dict, cfg, _runs_root, cancel_event=cancel_event,
            ):
                if frame.done:
                    payload = json.dumps({"stats": frame.stats or {}})
                    yield f"event: done\ndata: {payload}\n\n"
                elif frame.status is not None:
                    payload = json.dumps({"status": frame.status})
                    yield f"event: status\ndata: {payload}\n\n"
                else:
                    payload = json.dumps(
                        {
                            "index": frame.index,
                            "messages": frame.messages,
                            "raw_text": frame.raw_text,
                            "parsed": frame.parsed,
                        }
                    )
                    yield f"event: row\ndata: {payload}\n\n"
        except Exception as e:  # noqa: BLE001 — surfaced to UI as a single error event
            payload = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {payload}\n\n"
        finally:
            cancel_event.set()

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/runs/lr-finder")
def lr_finder(body: _LrFinderBody) -> dict:
    """Spawn N short mini-runs at different learning rates so the
    user can pick one before committing to a full training session.

    Each child run inherits the user's full RunConfig but overrides
    the learning rate, caps ``max_steps`` to the requested budget,
    and tags ``purpose="lr_finder"`` so the Library / main Runs view
    can opt to hide them from the headline list.

    Returns the list of created run ids in submission order; the
    frontend redirects to the Compare view with all of them
    pre-selected.
    """
    if not body.learning_rates:
        raise HTTPException(
            status_code=400,
            detail="learning_rates must contain at least one value.",
        )
    if len(body.learning_rates) > 6:
        # Bigger sweeps don't fit on the compare chart's color
        # palette and dilute the "pick the best one" intent.
        raise HTTPException(
            status_code=400,
            detail="learning_rates capped at 6 entries per finder run.",
        )
    for lr in body.learning_rates:
        if not math.isfinite(lr) or lr <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"Each learning rate must be a finite positive number; got {lr}.",
            )

    created_ids: list[str] = []
    try:
        for lr in body.learning_rates:
            new_cfg = body.config.model_copy(
                update={
                    "learning_rate": lr,
                    "max_steps": body.steps_per_run,
                    "purpose": "lr_finder",
                    # Ensure we don't accidentally chain a parent adapter
                    # into the sniff: each LR run starts fresh.
                    "resume_from": None,
                }
            )
            _validate_run_config(new_cfg)
            run = _store.create(new_cfg)
            created_ids.append(run.id)
    except HTTPException:
        # Roll back partially-created runs so a failure on (say) the
        # third LR doesn't leave two orphan PENDING runs littering the
        # Runs page. The rollback is best-effort — if rmtree fails for
        # one we still re-raise the original error.
        for rid in created_ids:
            try:
                _store.delete(rid)
            except Exception:  # noqa: BLE001 — never let cleanup mask the original failure
                pass
        raise

    # Drive each child to completion sequentially in a background
    # thread. Without this the runs would sit PENDING forever — the
    # normal create-then-stream path requires a foreground SSE
    # connection to consume the executor's generator. The user
    # doesn't want to babysit three SSE streams just for a sniff
    # test, so we drain them ourselves; events are persisted to
    # disk so the Compare view sees per-run progress as it polls.
    def _run_chain(ids: list[str]) -> None:
        for rid in ids:
            for _ in _executor.execute(rid):
                pass

    threading.Thread(
        target=_run_chain, args=(created_ids,), daemon=True,
    ).start()
    return {"run_ids": created_ids, "steps_per_run": body.steps_per_run}


@router.post("/runs")
def create_run(cfg: RunConfig) -> dict:
    _validate_run_config(cfg)
    run = _store.create(cfg)
    return {"id": run.id, "status": run.status.value}


def _adapter_size_bytes(run) -> int | None:
    """Compute the size on disk of the adapter weights for one run.

    The Library page surfaces this so users can see at a glance which
    runs are eating their disk. We return None for runs that don't have
    an adapter on disk yet (PENDING / RUNNING / FAILED before any
    save) — the UI renders that as "—" instead of "0 B".

    Walks the run dir for the known adapter filenames + every
    ``checkpoint-*/`` (HF Trainer's intermediate save layout). Sum is
    bytes; the UI formats it.
    """
    if run.output_dir is None:
        return None
    run_dir = Path(run.output_dir)
    if not run_dir.exists():
        return None
    candidates = [
        run_dir / "adapter_model.safetensors",
        run_dir / "adapters.safetensors",
    ]
    total = 0
    found_any = False
    for c in candidates:
        if c.exists():
            total += c.stat().st_size
            found_any = True
    for ckpt in run_dir.glob("checkpoint-*"):
        if ckpt.is_dir():
            for f in ckpt.glob("**/*"):
                if f.is_file():
                    total += f.stat().st_size
                    found_any = True
    return total if found_any else None


@router.get("/runs")
def list_runs() -> dict:
    """List every known run, newest first.

    For SUCCEEDED runs we also include ``adapter_size_bytes`` so the
    Library page doesn't need a second round-trip per row. Other
    statuses don't get the size because their on-disk artifacts are
    transient (mid-training checkpoints, partial GGUFs) and would
    just confuse the user if surfaced as "this is your adapter".
    """
    out = []
    for r in _store.list():
        d = r.model_dump(mode="json")
        if r.status == RunStatus.SUCCEEDED:
            d["adapter_size_bytes"] = _adapter_size_bytes(r)
        out.append(d)
    return {"runs": out}


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


class _GenerateBody(BaseModel):
    """Body for POST /runs/{id}/generate. Mirrors GenerationConfig
    but lets pydantic do the parsing so we get clean 422s on bad
    input shapes."""
    prompt: str = Field(min_length=1, max_length=8192)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)


@router.post("/runs/{run_id}/generate")
def generate_run(run_id: str, body: _GenerateBody) -> StreamingResponse:
    """Stream generated tokens from this run's adapter as SSE events.

    Each emitted token becomes a ``token`` event with ``{"text": "..."}``;
    the final frame is ``done`` with ``{}``. The frontend reads these
    via EventSource and appends ``text`` to the visible buffer.

    The actual model load happens lazily on first call and stays
    cached in process for subsequent calls — see inference.playground
    for the cache contract. Errors during load surface as a single
    ``error`` event so the UI can show a clean message instead of
    the SSE connection silently dying.
    """
    run = _get_succeeded_run_or_404(run_id, "inference")
    # VLM runs need image inputs and a Vision2Seq model class — the
    # current playground only handles text-in/text-out. Refuse upfront
    # rather than letting the model load itself the wrong way.
    if run.config.backend in _VLM_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Inference playground doesn't support vision-language runs "
                "yet (this run uses a VLM backend). Use the merged model "
                "directly via mlx_vlm/transformers for image+text inference."
            ),
        )
    cfg = GenerationConfig(
        prompt=body.prompt,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
    )
    run_dict = run.model_dump(mode="json")
    # Per-request cancel signal. Set by the gen()'s finally if the SSE
    # consumer disconnects (close-of-stream propagates GeneratorExit
    # through the yield). The HF backend's StoppingCriteria polls it
    # so model.generate stops at the next step instead of running out
    # the entire token budget when the user already navigated away.
    cancel_event = threading.Event()

    def gen():
        try:
            for tok in generate_stream(run_dict, cfg, _runs_root, cancel_event):
                if tok.done:
                    yield "event: done\ndata: {}\n\n"
                elif tok.status is not None:
                    payload = json.dumps({"status": tok.status})
                    yield f"event: status\ndata: {payload}\n\n"
                else:
                    payload = json.dumps({"text": tok.text})
                    yield f"event: token\ndata: {payload}\n\n"
        except Exception as e:  # noqa: BLE001 — surfaced to UI as a single error event
            payload = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {payload}\n\n"
        finally:
            # If gen() ends because the consumer abandoned mid-stream,
            # this runs before our local generator's stack unwinds.
            # Setting the event here lets the inference module's
            # finally blocks know to stop the model worker thread.
            cancel_event.set()

    return StreamingResponse(gen(), media_type="text/event-stream")


class _EvalBody(BaseModel):
    """POST body for the eval suite endpoint.

    ``prompts`` is bounded so a misclick can't queue a 1000-prompt
    run that takes hours. ``max_tokens`` is per-prompt, kept smaller
    than the playground default since users skim eval outputs.
    """
    prompts: list[str] = Field(default_factory=list, max_length=20)
    max_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


@router.get("/runs/{run_id}/eval/defaults")
def get_eval_defaults(run_id: str) -> dict:
    """Return the suggested default prompts for a run's model family.

    The Eval screen calls this on mount to pre-fill the textarea with
    something useful for THIS model rather than the generic placeholder
    set. Lookup is by family so a registry fork (different model_id,
    same family) inherits the curated defaults.
    """
    run = _get_run_or_404(run_id)
    # Find the registry entry — falls back to the generic prompts when
    # the user trained a custom HF id that isn't in our allowlist.
    match = next(
        (e for e in _registry.entries(include_restricted=True) if e.id == run.config.model_id),
        None,
    )
    family = match.family if match else None
    return {
        "family": family,
        "prompts": list(default_prompts_for_family(family)),
    }


@router.post("/runs/{run_id}/eval")
def eval_run(run_id: str, body: _EvalBody) -> StreamingResponse:
    """Stream side-by-side base/adapter outputs for a list of prompts.

    Wire format mirrors the generate endpoint:
      - ``status``: load progress / phase transitions
      - ``token``: ``{"role", "prompt_index", "text"}`` — append text
        to the (role, prompt_index) cell on the UI side
      - ``done``: end of suite
      - ``error``: surfaced from the inference layer

    Same VLM gate as /generate — eval suite is text-in/text-out only.
    """
    run = _get_succeeded_run_or_404(run_id, "eval")
    if run.config.backend in _VLM_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail="Eval suite doesn't support vision-language runs yet.",
        )

    # Sanitize: drop empty prompts, trim, and fall back to family-aware
    # defaults if the user submitted an empty list. Family lookup
    # mirrors the /eval/defaults endpoint so the two paths agree on
    # what gets run.
    cleaned = [p.strip() for p in (body.prompts or []) if p and p.strip()]
    if cleaned:
        prompts = tuple(cleaned)
    else:
        match = next(
            (e for e in _registry.entries(include_restricted=True) if e.id == run.config.model_id),
            None,
        )
        prompts = default_prompts_for_family(match.family if match else None)
    cfg = EvalConfig(
        prompts=prompts,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
    )
    run_dict = run.model_dump(mode="json")
    cancel_event = threading.Event()
    skip_event = threading.Event()
    _eval_skip_events[run_id] = skip_event

    def gen():
        try:
            for frame in evaluate(
                run_dict, cfg, _runs_root, cancel_event, skip_event=skip_event,
            ):
                if frame.done:
                    yield "event: done\ndata: {}\n\n"
                elif frame.status is not None:
                    payload = json.dumps({"status": frame.status})
                    yield f"event: status\ndata: {payload}\n\n"
                else:
                    payload = json.dumps({
                        "role": frame.role,
                        "prompt_index": frame.prompt_index,
                        "text": frame.text,
                    })
                    yield f"event: token\ndata: {payload}\n\n"
        except Exception as e:  # noqa: BLE001
            payload = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {payload}\n\n"
        finally:
            cancel_event.set()
            # Drop the skip flag — the eval is over, future eval
            # streams for this run will register their own. Leaving
            # it would let a stale "skip" fire on the next attempt.
            _eval_skip_events.pop(run_id, None)

    return StreamingResponse(gen(), media_type="text/event-stream")


class _ComparePromptsBody(BaseModel):
    """POST body for the A/B prompt comparator (F-A3).

    ``left_run_id`` / ``right_run_id`` reference SUCCEEDED runs in
    the store; either side may also be the synthetic ``base:<model_id>``
    id the eval suite uses to mean "load base only" — but the route
    rejects that here because A/B compare is intended for two real
    runs from the Library. (To compare a run against its own base,
    use the existing /runs/{id}/eval endpoint instead.)
    """

    left_run_id: str
    right_run_id: str
    prompts: list[str] = Field(default_factory=list, max_length=20)
    max_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


# Per-comparison skip flags. Keyed by ``(left_id, right_id)`` so a
# concurrent compare against a different pair doesn't cross-trigger.
_compare_skip_events: dict[tuple[str, str], threading.Event] = {}


@router.post("/compare/prompts")
def compare_prompts(body: _ComparePromptsBody) -> StreamingResponse:
    """Stream side-by-side outputs for a list of prompts against two
    runs. Wire format mirrors /eval but with roles ``left`` / ``right``
    instead of ``base`` / ``adapter``.

    Pre-flight rejects VLM backends and refuses to compare two runs on
    different base models — same scorers wouldn't make sense across
    different tokenisers and the user's prompt set is unlikely to
    exercise both fairly.
    """
    from llm_chain_sidecar.inference import eval_suite as _eval

    left = _get_succeeded_run_or_404(body.left_run_id, "compare")
    right = _get_succeeded_run_or_404(body.right_run_id, "compare")
    if left.id == right.id:
        raise HTTPException(
            status_code=400,
            detail="Pick two different runs — comparing a run against itself is a no-op.",
        )
    for run, side in ((left, "left"), (right, "right")):
        if run.config.backend in _VLM_BACKENDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Compare doesn't support vision-language runs "
                    f"({side} run uses {run.config.backend!r})."
                ),
            )
    if left.config.model_id != right.config.model_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Compare needs two runs that share the same base model "
                f"({left.config.model_id!r} vs {right.config.model_id!r}). "
                "Otherwise tokenizers differ and side-by-side scoring is "
                "apples-to-oranges."
            ),
        )

    cleaned = [p.strip() for p in (body.prompts or []) if p and p.strip()]
    if cleaned:
        prompts = tuple(cleaned)
    else:
        match = next(
            (
                e
                for e in _registry.entries(include_restricted=True)
                if e.id == left.config.model_id
            ),
            None,
        )
        prompts = default_prompts_for_family(match.family if match else None)

    cfg = _eval.EvalConfig(
        prompts=prompts,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
    )
    cancel_event = threading.Event()
    skip_event = threading.Event()
    pair_key = (left.id, right.id)
    _compare_skip_events[pair_key] = skip_event

    left_dict = left.model_dump(mode="json")
    right_dict = right.model_dump(mode="json")

    def gen():
        try:
            for frame in _eval.compare_pairwise(
                left_dict,
                right_dict,
                cfg,
                _runs_root,
                cancel_event=cancel_event,
                skip_event=skip_event,
            ):
                if frame.done:
                    yield "event: done\ndata: {}\n\n"
                elif frame.status is not None:
                    payload = json.dumps({"status": frame.status})
                    yield f"event: status\ndata: {payload}\n\n"
                else:
                    payload = json.dumps(
                        {
                            "role": frame.role,
                            "prompt_index": frame.prompt_index,
                            "text": frame.text,
                        }
                    )
                    yield f"event: token\ndata: {payload}\n\n"
        except Exception as e:  # noqa: BLE001
            payload = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {payload}\n\n"
        finally:
            cancel_event.set()
            _compare_skip_events.pop(pair_key, None)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/compare/skip")
def skip_compare_prompt(left_run_id: str, right_run_id: str) -> dict:
    """Skip the current prompt in an in-flight A/B compare.

    Mirrors /eval/skip's contract: 409 when no compare is running for
    the (left, right) pair so a stale Skip click after the suite
    finishes is a soft no-op rather than a hard error.
    """
    _validate_run_id(left_run_id)
    _validate_run_id(right_run_id)
    ev = _compare_skip_events.get((left_run_id, right_run_id))
    if ev is None:
        raise HTTPException(
            status_code=409,
            detail="No compare is running for that pair.",
        )
    ev.set()
    return {"signaled": True}


@router.post("/runs/{run_id}/eval/skip")
def skip_eval_prompt(run_id: str) -> dict:
    """Tell the in-flight eval to abandon the current prompt and move
    on. No-op (with 409) when there's no eval running for this run —
    a stale "Skip" click after the suite finished shouldn't crash
    anything, just inform the user the suite is already done.
    """
    _validate_run_id(run_id)
    ev = _eval_skip_events.get(run_id)
    if ev is None:
        raise HTTPException(
            status_code=409,
            detail="No eval is running for this run.",
        )
    ev.set()
    return {"signaled": True}


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    _validate_run_id(run_id)
    if not _executor.cancel(run_id):
        # Either the run never started streaming (no in-flight executor) or it
        # has already finished. 409 communicates "no active run to cancel".
        raise HTTPException(status_code=409, detail="run not active")
    return {"canceled": True}


class _ResumeBody(BaseModel):
    """Body for POST /runs/{id}/resume.

    The caller picks the additional epochs and (optionally) a fresh
    learning rate; everything else is inherited from the parent so the
    LoRA shapes line up. The endpoint returns the same shape as
    ``create_run`` so the frontend can redirect to the new run id.
    """
    epochs: int = 1
    learning_rate: float | None = None


@router.post("/runs/{run_id}/resume")
def resume_run(run_id: str, body: _ResumeBody) -> dict:
    """Spawn a new run that continues from this one's adapter.

    The new run inherits everything from the parent except epochs (the
    caller decides how much more to train) and optionally the learning
    rate (a smaller LR is often appropriate for continuation). The
    parent run is unchanged — its adapter stays on disk in case the
    user wants to compare or branch later.
    """
    parent = _get_succeeded_run_or_404(run_id, "resume")
    if body.epochs <= 0:
        raise HTTPException(status_code=400, detail="epochs must be >= 1.")
    new_cfg = parent.config.model_copy(
        update={
            "epochs": body.epochs,
            "learning_rate": (
                body.learning_rate
                if body.learning_rate is not None
                else parent.config.learning_rate
            ),
            "resume_from": run_id,
        }
    )
    _validate_run_config(new_cfg)
    new_run = _store.create(new_cfg)
    return {"id": new_run.id, "status": new_run.status.value}


_CLEANABLE_STATUSES = {"failed", "canceled", "succeeded"}


class _CleanupBody(BaseModel):
    """POST body for the cleanup endpoint.

    ``older_than_days`` of 0 means "anything that matches the status
    filter", which is what an "Apply now to everything that's
    failed" sweep wants. Negative values are nonsense and rejected.

    ``statuses`` controls which terminal states get swept. Active
    runs (PENDING / RUNNING) are NEVER deletable through this
    endpoint — same rule as DELETE /runs/{id}.
    """
    older_than_days: float = Field(default=0, ge=0)
    statuses: list[str] = Field(default_factory=lambda: ["failed", "canceled"])


@router.post("/maintenance/cleanup")
def cleanup_runs(body: _CleanupBody) -> dict:
    """Bulk-delete terminal-state runs matching the policy.

    Used by Settings → "Apply now" and (eventually) by the startup
    sweep. We compute the cutoff client-side from the user's request
    rather than from server time so re-running a sweep is idempotent
    relative to the user's stated intent: "delete failed runs older
    than 7 days" produces the same set on retry.
    """
    from llm_chain_sidecar import inference as _inference

    bad = [s for s in body.statuses if s not in _CLEANABLE_STATUSES]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown statuses {bad}. Allowed: "
                f"{sorted(_CLEANABLE_STATUSES)}."
            ),
        )

    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=body.older_than_days)
    targets = [
        r for r in _store.list()
        if r.status.value in body.statuses and r.created_at < cutoff
    ]
    cached = _inference.cached_run_id()
    deleted_ids: list[str] = []
    freed_bytes = 0
    for r in targets:
        size = _adapter_size_bytes(r) or 0
        try:
            _store.delete(r.id)
        except FileNotFoundError:
            # Run dir already gone (e.g. user manually deleted out
            # from under us). Don't fail the whole sweep over it.
            continue
        deleted_ids.append(r.id)
        freed_bytes += size
        # If the inference cache held this run, drop it — same logic
        # as DELETE /runs/{id}, applied per-target.
        if cached == r.id:
            _inference.free_cache()
    return {
        "deleted_count": len(deleted_ids),
        "freed_bytes": freed_bytes,
        "deleted_ids": deleted_ids,
    }


@router.delete("/runs/{run_id}")
def delete_run(run_id: str) -> dict:
    """Remove a run from disk.

    Refuses while the run is in flight — deleting the dir out from
    under a running trainer would have it failing to write events or
    save adapters mid-step. The user has to Cancel first; once the
    run reaches a terminal state (succeeded / failed / canceled) it's
    safe to delete.
    """
    from llm_chain_sidecar import inference as _inference

    run = _get_run_or_404(run_id)
    if run.status in (RunStatus.PENDING, RunStatus.RUNNING):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run is {run.status.value}. Cancel it first, then delete "
                "once it's reached a terminal state."
            ),
        )
    _store.delete(run_id)
    # If the playground had this run's model warm in the cache, the
    # model object holds tensors that no longer correspond to anything
    # on disk. Drop it so the memory comes back and the next /generate
    # against a different run starts cleanly.
    if _inference.cached_run_id() == run_id:
        _inference.free_cache()
    return {"deleted": True}


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


def _hub_state_path(run_id: str) -> Path:
    return _runs_root / run_id / _HUB_STATE_FILE


def _read_hub_state(run_id: str) -> dict | None:
    p = _hub_state_path(run_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _write_hub_state(run_id: str, state: dict) -> None:
    """Atomic write — same rationale as _write_gguf_state."""
    target = _hub_state_path(run_id)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, target)


def _run_hub_push(run_id: str, repo_id: str, private: bool, folder: str) -> None:
    """Background worker. Mirrors _run_gguf_export's state-file pattern so
    the UI can poll progress instead of staring at a frozen 'Pushing…'
    spinner during multi-minute uploads.

    Each tqdm progress line that huggingface_hub emits during upload
    becomes a ``latest_log`` update on the state file. The terminal state
    is either ``done`` (with ``url``) or ``failed`` (with ``error``).
    """
    def _set_state(**fields) -> None:
        current = _read_hub_state(run_id) or {}
        current.update(fields)
        _write_hub_state(run_id, current)

    def _on_progress(line: str) -> None:
        _set_state(latest_log=line)

    try:
        _set_state(
            status="running", repo_id=repo_id, private=private, folder=folder,
            latest_log=None,
        )
        url = exports.push_to_hub(
            run_id, repo_id, runs_root=_runs_root, private=private, folder=folder,
            on_progress=_on_progress,
        )
        _set_state(status="done", url=url, latest_log=None)
    except exports.HubAuthError as e:
        _set_state(status="failed", error=str(e), error_kind="auth", latest_log=None)
    except FileNotFoundError as e:
        _set_state(status="failed", error=str(e), error_kind="missing", latest_log=None)
    except ValueError as e:
        _set_state(status="failed", error=str(e), error_kind="invalid", latest_log=None)
    except Exception as e:  # noqa: BLE001 — surface anything else verbatim
        _set_state(status="failed", error=str(e), error_kind="unknown", latest_log=None)


@router.post("/runs/{run_id}/export/hub")
def push_run_to_hub(run_id: str, body: _HubPushBody) -> JSONResponse:
    """Kick off a background hub upload. Returns 202 immediately with the
    initial state; the UI polls GET /export/hub for progress + result.

    Pre-flight checks (auth, repo_id well-formedness, folder validity)
    are cheap, so we run them synchronously first and return their
    failure as 4xx without spawning the worker. Only the actual
    network upload runs in the background.
    """
    _get_succeeded_run_or_404(run_id, "hub push")
    if not exports.is_hf_signed_in():
        # Mirror the previous synchronous contract: 401 lets the UI
        # prompt the user to run huggingface-cli login.
        raise HTTPException(
            status_code=401,
            detail=(
                "Not signed in to Hugging Face. "
                "Run `huggingface-cli login` in a terminal and try again."
            ),
        )
    if body.folder not in ("adapter", "merged"):
        raise HTTPException(
            status_code=400,
            detail=f"unknown folder: {body.folder!r}; pick 'adapter' or 'merged'",
        )

    state = _read_hub_state(run_id)
    if state and state.get("status") == "running":
        return JSONResponse(status_code=202, content=state)

    threading.Thread(
        target=_run_hub_push,
        args=(run_id, body.repo_id, body.private, body.folder),
        daemon=True,
    ).start()
    initial = {
        "status": "running",
        "repo_id": body.repo_id,
        "private": body.private,
        "folder": body.folder,
    }
    return JSONResponse(status_code=202, content=initial)


@router.get("/runs/{run_id}/export/hub")
def get_hub_push(run_id: str) -> dict:
    _get_run_or_404(run_id)
    state = _read_hub_state(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="no hub push started for this run")
    return state
