"""Push a trained adapter (or its merged GGUF) up to the Hugging Face Hub.

Auth: we read the token via huggingface_hub's own resolver, which checks the
``HF_TOKEN`` env var, the legacy ``HUGGING_FACE_HUB_TOKEN``, and the file at
``~/.cache/huggingface/token`` written by ``huggingface-cli login``. We never
prompt or store the token ourselves.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

# Optional progress callback. The route layer wires this to a state file
# the UI polls so multi-minute uploads show progress instead of a frozen
# spinner. Receives one tqdm-style line at a time.
ProgressCb = Callable[[str], None]


class HubAuthError(RuntimeError):
    """Raised when no HF token is available so callers can surface a clean
    'not signed in' message instead of a stack trace."""


def _resolve_token() -> str | None:
    """Best-effort lookup of the HF token. Returns None when the user hasn't
    run ``huggingface-cli login`` and hasn't exported HF_TOKEN."""
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
    except Exception:
        token = None
    if token:
        return token
    import os
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


@contextmanager
def _emit_upload_progress(on_progress: ProgressCb | None):
    """Patch ``tqdm.auto.tqdm.display`` for the duration of an HfApi
    upload so each progress bar update becomes a callback invocation.

    Mirrors trainers/hf_progress.py's download patcher — same gotcha
    (huggingface_hub's tqdm has no public per-bar callback) and the
    same single-process safety: this is the only place we patch tqdm
    during an upload, and the route layer ensures only one upload runs
    at a time per process.
    """
    if on_progress is None:
        yield
        return
    import tqdm.auto

    original_display = tqdm.auto.tqdm.display
    last_emit_pct: dict[int, float] = {}

    def patched_display(self, msg=None, pos=None):
        total = getattr(self, "total", None)
        n = getattr(self, "n", 0) or 0
        desc = getattr(self, "desc", "") or ""
        if total and total > 0:
            pct = (n / total) * 100
            key = id(self)
            prev = last_emit_pct.get(key, -1.0)
            # Throttle to 1% deltas to avoid saturating the state file
            # writer with every byte. Always emit the final tick so
            # the UI shows 100% before the bar disappears.
            if pct - prev >= 1.0 or n >= total:
                last_emit_pct[key] = pct
                try:
                    on_progress(f"{desc} {n}/{total} ({pct:.0f}%)")
                except Exception:  # noqa: BLE001 — never let UI hook kill the upload
                    pass
        return original_display(self, msg, pos)

    tqdm.auto.tqdm.display = patched_display
    try:
        yield
    finally:
        tqdm.auto.tqdm.display = original_display


def push_to_hub(
    run_id: str,
    repo_id: str,
    runs_root: Path,
    private: bool = True,
    folder: str = "adapter",
    on_progress: ProgressCb | None = None,
) -> str:
    """Push a run's output to ``repo_id`` and return the resolved URL.

    Args:
        run_id: the LLM-Chain run id; we look up its output dir under runs_root.
        repo_id: Hugging Face repo, e.g. ``user/my-adapter``.
        runs_root: root the sidecar configured for runs storage.
        private: create the repo private if it doesn't exist (default).
        folder: which subfolder under the run dir to upload —
            ``"adapter"`` uploads the run dir as-is (the LoRA checkpoint),
            ``"merged"`` uploads the merged HF dir from a prior GGUF export.

    Raises:
        HubAuthError: when no HF token can be resolved.
        FileNotFoundError: when the run dir or sub-folder doesn't exist.
    """
    token = _resolve_token()
    if not token:
        raise HubAuthError(
            "Not signed in to Hugging Face. "
            "Run `huggingface-cli login` in a terminal and try again."
        )

    run_dir = runs_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run {run_id} not found at {run_dir}")

    if folder == "adapter":
        upload_dir = run_dir
    elif folder == "merged":
        upload_dir = run_dir / "merged"
    else:
        raise ValueError(f"unknown folder: {folder!r}; pick 'adapter' or 'merged'")
    if not upload_dir.exists():
        raise FileNotFoundError(f"{upload_dir} doesn't exist; nothing to upload")

    # Defense in depth: if the run somehow reached SUCCEEDED without the
    # trainer writing a weights file (a previous-version artifact, a manual
    # edit, etc.), the upload patterns would strip everything except a
    # bare run.json (already in ignore_patterns) and create an empty repo
    # on HF. Better to refuse with a clear message.
    if folder == "adapter":
        has_weights = any(
            (upload_dir / name).exists()
            for name in ("adapter_model.safetensors", "adapters.safetensors")
        ) or any(upload_dir.glob("checkpoint-*/adapter_model.safetensors"))
        if not has_weights:
            raise FileNotFoundError(
                f"No adapter weights found under {upload_dir}. The run "
                "succeeded but produced no adapter file — re-train and "
                "push the new run."
            )
    else:  # merged
        if not (upload_dir / "config.json").exists():
            raise FileNotFoundError(
                f"No merged model config found at {upload_dir / 'config.json'}. "
                "Run the GGUF export (or its merge step) first."
            )

    # Lazy import — keeps module import cheap for unrelated request paths.
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    with _emit_upload_progress(on_progress):
        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            repo_type="model",
            # Defense-in-depth exclude list. Anything that could leak the
            # user's local context — dataset path, training logs, raw
            # inputs, partial GGUFs — is filtered before HF sees the bytes:
            # - run.json carries dataset_path, which is often inside
            #   ~/Documents or a user's home directory; uploading it to a
            #   public repo would doxx the user.
            # - events.jsonl is the per-step log; not sensitive but bloats
            #   the repo to no useful purpose.
            # - _mlx_data/_mlx_vlm_data hold the staged training rows.
            # - checkpoint-*/** is HF Trainer's intermediate save layout
            #   (the previous "checkpoints/**" pattern never matched the
            #   real dirs).
            # - *.partial guards against an interrupted GGUF export
            #   polluting the publish.
            ignore_patterns=[
                "*.jsonl",
                "*.csv",
                "run.json",
                "events.jsonl",
                "export-gguf.json",
                "export-hub.json",
                "_mlx_data/**",
                "_mlx_vlm_data/**",
                "checkpoint-*/**",
                "checkpoints/**",
                "*.partial",
                ".git/**",
            ],
        )
    return f"https://huggingface.co/{repo_id}"


def is_hf_signed_in() -> bool:
    """Lightweight check the UI uses before showing the push form."""
    return _resolve_token() is not None
