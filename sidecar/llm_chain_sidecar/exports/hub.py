"""Push a trained adapter (or its merged GGUF) up to the Hugging Face Hub.

Auth: we read the token via huggingface_hub's own resolver, which checks the
``HF_TOKEN`` env var, the legacy ``HUGGING_FACE_HUB_TOKEN``, and the file at
``~/.cache/huggingface/token`` written by ``huggingface-cli login``. We never
prompt or store the token ourselves.
"""
from __future__ import annotations

import json
from pathlib import Path


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


def push_to_hub(
    run_id: str,
    repo_id: str,
    runs_root: Path,
    private: bool = True,
    folder: str = "adapter",
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

    # Lazy import — keeps module import cheap for unrelated request paths.
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=repo_id,
        repo_type="model",
        # Limit to weights + tokenizer + readme; keep raw datasets out of the
        # publish so accidentally-trained-on private data doesn't leak.
        ignore_patterns=["*.jsonl", "*.csv", "checkpoints/**", ".git/**"],
    )
    return f"https://huggingface.co/{repo_id}"


def is_hf_signed_in() -> bool:
    """Lightweight check the UI uses before showing the push form."""
    return _resolve_token() is not None
