"""Shared row → text helpers for the HF-based trainers.

Both CpuTrainer and HfCudaTrainer feed a HuggingFace ``Dataset`` of
``{"text": str}`` rows into the tokenizer. The way each loader-format row
gets flattened into that string used to live inline in both trainers, so any
fix had to be made twice and the chat path silently bypassed the tokenizer's
chat template. This module owns the conversion and the pad-token fallback.

Also exposes ``run_in_background_with_sentinel``, a small wrapper used by
all three HF trainers to start the HF Trainer in a daemon thread without
the historical deadlock when ``train()`` raises before its on_train_end
callback runs.
"""
from __future__ import annotations

import queue
from collections.abc import Iterator
from threading import Thread
from typing import Callable

from llm_chain_sidecar.datasets.types import DatasetFormat


def run_in_background_with_sentinel(
    target: Callable[[], None], events: "queue.Queue[dict | None]"
) -> Thread:
    """Run ``target`` in a daemon thread; guarantee a None sentinel reaches
    ``events`` regardless of how target exits.

    The HF ``Trainer.train()`` only fires its ``on_train_end`` callback —
    which is the natural producer of the ``None`` sentinel — on a clean
    finish. If the worker raises (CUDA OOM, dataset bug, model.forward
    error) the callback never fires and the consumer blocks on
    ``events.get()`` forever. Wrapping the worker so it always pushes a
    sentinel turns "consumer hangs" into "consumer sees an error event"
    — much easier for the user to understand and recover from.
    """
    def _runner() -> None:
        try:
            target()
        except BaseException as e:  # noqa: BLE001 — re-raised on consumer side
            events.put({"type": "error", "exception": e})
        finally:
            events.put(None)

    t = Thread(target=_runner, daemon=True)
    t.start()
    return t


def make_event_callback(events: "queue.Queue[dict | None]", cancel_event):
    """Build the HF TrainerCallback we attach to every HF-trained run.

    Three responsibilities, all wired identically across the three
    HF-style trainers (CPU / CUDA / VLM): forward each loss-bearing log
    onto ``events`` as a ``step`` dict, observe ``cancel_event`` at every
    step boundary so a cancel POST takes effect within one step, and
    emit the ``None`` sentinel from ``on_train_end`` so the consumer's
    queue loop terminates on natural completion.
    """
    # Lazy import: this module is imported by the API layer at startup,
    # but ``transformers`` is a heavy import we only need when actually
    # training.
    from transformers import TrainerCallback

    class _EventForwarderCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                events.put({
                    "type": "step",
                    "step": state.global_step,
                    "total_steps": state.max_steps,
                    "loss": logs["loss"],
                    "lr": logs.get("learning_rate"),
                })

        def on_step_end(self, args, state, control, **kw):
            if cancel_event.is_set():
                control.should_training_stop = True

        def on_train_end(self, args, state, control, **kw):
            events.put(None)

    return _EventForwarderCallback()


def resume_adapter_dir(output_dir: str, resume_from: str | None) -> "str | None":
    """Resolve the parent run's adapter directory for HF-trainer resume.

    HF-saved adapters live at ``<run_dir>/adapter_model.safetensors``
    next to ``adapter_config.json``; pass the directory itself to
    ``PeftModel.from_pretrained``. Returns None only when
    ``resume_from`` is unset.

    When ``resume_from`` IS set but no adapter is found on disk (parent
    deleted between create-time validation and trainer execution),
    raise ``FileNotFoundError``. Falling back to fresh init would
    silently train without the parent's weights — a meaningful
    intent-mismatch the trainer should surface as an ERROR event.
    """
    if not resume_from:
        return None
    from pathlib import Path as _Path

    parent = _Path(output_dir).parent / resume_from
    if (parent / "adapter_model.safetensors").exists():
        return str(parent)
    # HF Trainer also lays out checkpoint-N/ subdirs; pick the highest
    # one if the run dir itself doesn't carry the adapter.
    checkpoints = sorted(
        (p for p in parent.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-", 1)[1]) if p.name.split("-", 1)[1].isdigit() else -1,
    )
    if checkpoints and (checkpoints[-1] / "adapter_model.safetensors").exists():
        return str(checkpoints[-1])
    raise FileNotFoundError(
        f"Cannot resume from run {resume_from}: no adapter file found "
        f"under {parent}. The parent run may have been deleted; start a "
        "fresh run or restore the parent."
    )


def pump_queue_until_sentinel(
    events: "queue.Queue[dict | None]",
) -> "Iterator[dict]":
    """Yield events from the queue until the sentinel arrives.

    ``None`` is the natural-end sentinel from ``on_train_end``. An entry
    of shape ``{"type": "error", "exception": ...}`` is the
    abnormal-exit sentinel placed by ``run_in_background_with_sentinel``
    when the worker raised — we re-raise it on the consumer side so the
    surrounding trainer's ``except`` block can convert it into a single
    ERROR TrainingEvent.
    """
    while True:
        ev = events.get()
        if ev is None:
            return
        if ev.get("type") == "error":
            raise ev["exception"]
        yield ev


def ensure_pad_token(tok) -> bool:
    """Tokenizers without a pad token fail ``padding="max_length"`` with a
    misleading 'Asking to pad but the tokenizer does not have a padding
    token' deep in collation. EOS is the standard fallback. If neither exists
    (rare — Pythia has one, but a custom tokenizer might not) fall through
    to the literal string ``[PAD]`` so we never silently leave pad_token
    unset.

    Returns True iff a brand new token was added to the vocabulary (the
    last-resort branch). Callers that load the model after the tokenizer
    must check this and call ``model.resize_token_embeddings(len(tok))``;
    otherwise the model's embedding matrix stays at its original vocab
    size and any input id pointing at the new pad token is out-of-bounds
    when the forward pass runs.
    """
    if tok.pad_token is not None:
        return False
    if tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        return False
    if getattr(tok, "unk_token", None):
        tok.pad_token = tok.unk_token
        return False
    tok.add_special_tokens({"pad_token": "[PAD]"})
    return True


def row_to_text(row: dict, ds_format: DatasetFormat, tok, text_column: str | None) -> str:
    """Convert a loader row into the training string for an HF causal LM.

    For chat-format datasets we go through ``tok.apply_chat_template`` so the
    model actually sees the format it was trained on (special tokens,
    role markers, BOS/EOS). The previous implementation joined messages with
    ``f"{role}: {content}"`` which silently bypassed the template and
    produced bad fine-tunes even on chat-capable models. If the tokenizer
    has no chat template we raise a clear error pointing at the model
    selection — matches what the route-level validator catches.
    """
    if ds_format == DatasetFormat.JSONL_CHAT:
        if not getattr(tok, "chat_template", None):
            raise ValueError(
                "Tokenizer has no chat_template — pick a chat-tuned variant "
                "of this model (e.g. an Instruct / Chat checkpoint) or use a "
                "non-chat dataset format (CSV, text-dir, HF Hub)."
            )
        return tok.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
    if ds_format == DatasetFormat.JSONL_CHAT_VISION:
        # Vision rows belong on the VLM trainers. Defense in depth: if a
        # bad backend resolution lands a vision row here, fail fast instead
        # of training on `[{'type': 'text', 'text': ...}]` literals.
        raise ValueError(
            "Vision dataset rows reached the text trainer — pick a vision "
            "model + cuda_vlm/mlx_vlm backend."
        )
    if ds_format == DatasetFormat.CSV:
        col = text_column or "text"
        if col not in row:
            raise ValueError(
                f"CSV column '{col}' not found in row keys {sorted(row)}"
            )
        return str(row[col])
    if ds_format == DatasetFormat.TEXT_DIR:
        return str(row.get("text", ""))
    if ds_format == DatasetFormat.HF_HUB:
        for col in (text_column, "text", "content", "input"):
            if col and col in row:
                return str(row[col])
        raise ValueError(
            f"Couldn't find a text column in HF Hub row {sorted(row)}; "
            "set 'text_column' on the dataset to point at the right field."
        )
    raise NotImplementedError(f"Trainer doesn't handle {ds_format}")
