"""Side-by-side eval: generate the same prompts against the base
model and the run's trained adapter, stream both outputs back to the
UI as they accumulate.

Why this matters: training loss going down is a *necessary* signal but
not a *sufficient* one. The user wants to see, qualitatively, that
prompts they care about behave differently after the fine-tune. This
module wires that.

Implementation reuses the playground module's per-backend streaming
(``_stream_mlx`` / ``_stream_hf``) and cache. We load the base
without the adapter, generate every prompt's "before" output, then
load the adapter and generate every prompt's "after". Two model
loads instead of two-per-prompt.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import playground


@dataclass(frozen=True)
class EvalConfig:
    """User-supplied parameters for one eval pass."""
    prompts: tuple[str, ...]
    max_tokens: int = 128
    temperature: float = 0.3  # lower than playground's 0.7 — eval wants
                              # deterministic-ish so before/after differences
                              # come from training, not sampling noise.


@dataclass(frozen=True)
class EvalFrame:
    """One streamed update from the eval orchestrator.

    Three event types share the same shape:
      - ``token``: ``role`` ("base" / "adapter"), ``prompt_index``,
        ``text`` is the decoded delta to append in the right column
        of the right row.
      - ``status``: ``status`` is informational ("Loading base…").
      - ``done``: ``done=True`` when the entire suite is finished.
    """
    role: str = ""           # "base" or "adapter" or "" for status/done
    prompt_index: int = -1   # which prompt in the suite this delta belongs to
    text: str = ""
    status: str | None = None
    done: bool = False


# Default prompt set — used when the caller doesn't pass any. Kept
# small (3 prompts) so a 7B model on Apple Silicon finishes in a
# couple of minutes total. A real user would override per project.
DEFAULT_PROMPTS = (
    "Hello, who are you?",
    "Write a one-sentence summary of what you can help me with.",
    "Translate this to French: 'Good morning, how are you today?'",
)


def evaluate(
    run_dict: dict[str, Any],
    cfg: EvalConfig,
    runs_root: Path,
    cancel_event: threading.Event | None = None,
) -> Iterator[EvalFrame]:
    """Yield streamed before/after outputs for each prompt.

    Sequencing:
      1. Load base only (no adapter). For each prompt, stream its
         "before" output frame-by-frame.
      2. Swap to base + adapter. For each prompt, stream "after".
      3. Yield ``done``.

    Status frames bracket each load so the UI can show "Loading base
    model…" / "Switching to fine-tuned adapter…" before the next
    burst of token frames arrives.

    Cancellation: a set ``cancel_event`` short-circuits the current
    prompt and skips remaining ones, then yields ``done``.
    """
    if not cfg.prompts:
        yield EvalFrame(done=True)
        return

    # The playground module's cache holds at most one model at a
    # time, keyed on (run_id, backend). We model "base only" as a
    # synthetic run id so the cache can hold base alongside adapter
    # without trampling each other across the suite.
    base_run = _base_only_run_dict(run_dict)
    adapter_run = run_dict

    yield EvalFrame(status="Loading base model (no adapter)…")
    yield from _stream_phase(
        run_dict=base_run,
        role="base",
        cfg=cfg,
        runs_root=runs_root,
        cancel_event=cancel_event,
    )

    if cancel_event is not None and cancel_event.is_set():
        yield EvalFrame(done=True)
        return

    yield EvalFrame(status="Loading adapter on top of base…")
    yield from _stream_phase(
        run_dict=adapter_run,
        role="adapter",
        cfg=cfg,
        runs_root=runs_root,
        cancel_event=cancel_event,
    )

    yield EvalFrame(done=True)


def _stream_phase(
    run_dict: dict[str, Any],
    role: str,
    cfg: EvalConfig,
    runs_root: Path,
    cancel_event: threading.Event | None,
) -> Iterator[EvalFrame]:
    """Generate every prompt under one model configuration (base or
    adapter), tagging each frame with its prompt index + role.

    We translate the playground's GenerationToken stream into
    EvalFrames so the orchestrator's wire format stays uniform —
    the route's SSE serializer doesn't have to special-case base vs
    adapter or worry about per-prompt indexing.
    """
    for idx, prompt in enumerate(cfg.prompts):
        if cancel_event is not None and cancel_event.is_set():
            return
        gcfg = playground.GenerationConfig(
            prompt=prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        for tok in playground.generate_stream(
            run_dict, gcfg, runs_root, cancel_event=cancel_event,
        ):
            if tok.done:
                # Per-prompt completion is implicit (next prompt starts
                # streaming under the same role). We don't forward an
                # internal "done" — only the suite-level done at the
                # end matters to the UI.
                continue
            if tok.status is not None:
                # The playground emits a "Loading model…" hint on its
                # cache miss; absorb it into our outer status.
                continue
            yield EvalFrame(
                role=role,
                prompt_index=idx,
                text=tok.text,
            )


def _base_only_run_dict(run_dict: dict[str, Any]) -> dict[str, Any]:
    """Build a synthetic run_dict that asks the playground loader to
    load the base model without applying the adapter.

    The playground keys its cache by ``run_id``, so we use a
    distinct id (``base:<model_id>``) that's stable across
    invocations. ``output_dir`` is empty — the loader uses it for
    adapter resolution; an empty path makes the HF/MLX loaders
    skip the adapter step.
    """
    cfg = run_dict["config"]
    base_id = f"base:{cfg['model_id']}"
    return {
        "id": base_id,
        "config": cfg,
        "output_dir": "",  # signals "no adapter" to the loader
    }
