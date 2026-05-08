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


# Generic fallback prompt set — used when neither the caller passed
# prompts nor the model belongs to a family with a curated default
# block. Kept small (3 prompts) so a 7B model on Apple Silicon
# finishes in a couple of minutes total. A real user would override
# per project.
DEFAULT_PROMPTS = (
    "Hello, who are you?",
    "Write a one-sentence summary of what you can help me with.",
    "Translate this to French: 'Good morning, how are you today?'",
)


# Family-aware default prompt sets. Each family gets prompts that
# exercise its typical fine-tune target — instruction-following models
# get a reasoning + summarisation mix, code-tuned variants get a
# code task, and so on. Keys match the ``family`` field on
# ``ModelEntry`` in the registry.
#
# The point of these defaults isn't that every user wants exactly
# these prompts — it's that opening Eval for the first time should
# show *something meaningful for THIS model* instead of the generic
# "Hello, who are you?" placeholder.
FAMILY_PROMPTS: dict[str, tuple[str, ...]] = {
    "Qwen3": (
        "Briefly explain what reinforcement learning from human feedback is.",
        "Summarise this sentence in 5 words: 'The committee unanimously decided to postpone the launch until safety reviews finish.'",
        "Translate to Spanish: 'I'd like to schedule a meeting for next Tuesday afternoon.'",
        "What's the next number in the sequence 2, 6, 12, 20, 30?",
    ),
    "Qwen2-VL": (
        # Vision runs are gated out of eval for now, but if a future
        # version unlocks them this is what we'd ship.
        "Describe what you see in detail.",
        "What text appears in this image?",
    ),
    "SmolLM": (
        "Hi! What can you help me with today?",
        "Write a one-sentence bedtime story about a robot who learned to dance.",
        "Translate 'thank you very much' into Japanese.",
    ),
    "Phi": (
        "What are three benefits of using transformers for NLP tasks?",
        "Solve step by step: if a train travels 60 miles in 90 minutes, what's its average speed in mph?",
        "Convert this to a polite email opener: 'I want to ask about my order'.",
    ),
    "Mistral": (
        "Briefly compare LoRA and full fine-tuning.",
        "Write a haiku about a thunderstorm.",
        "What's the time complexity of binary search and why?",
    ),
    "TinyLlama": (
        "Hi! Tell me one fun fact about octopuses.",
        "Write a one-sentence motivational quote.",
        "What's 17 × 24?",
    ),
    "Pythia": (
        # Base models without a chat template — ship completion-style
        # prompts so the model can continue them naturally.
        "The benefits of open-source language models include",
        "Once upon a time in a small village,",
        "Q: What is the capital of France?\nA:",
    ),
    "OLMo": (
        "Explain the difference between a list and a tuple in Python.",
        "Write a short paragraph describing a sunset over the ocean.",
        "What is gradient descent?",
    ),
    "Llama": (
        "What are some practical applications of LLMs in healthcare?",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "Write a polite reminder email about a missed deadline.",
    ),
    "Gemma": (
        "What are three ways to improve writing clarity?",
        "Explain quantum entanglement to a curious 12-year-old.",
        "Write a short product description for a noise-cancelling headphone.",
    ),
    "DeepSeek": (
        "Walk me through implementing FizzBuzz in Python.",
        "What's the difference between supervised and self-supervised learning?",
        "Solve: a rectangle has area 48 and perimeter 28. What are its dimensions?",
    ),
}


def default_prompts_for_family(family: str | None) -> tuple[str, ...]:
    """Return the prompt set the Eval screen pre-fills for a model.

    Looks up by ``family`` (e.g. ``"Qwen3"``, ``"SmolLM"``) so a fork
    of a base model — same family, different model_id — still gets
    the curated defaults. Falls back to the generic
    ``DEFAULT_PROMPTS`` for unknown families so newly-added registry
    entries don't surface as an empty list.
    """
    if family and family in FAMILY_PROMPTS:
        return FAMILY_PROMPTS[family]
    return DEFAULT_PROMPTS


def evaluate(
    run_dict: dict[str, Any],
    cfg: EvalConfig,
    runs_root: Path,
    cancel_event: threading.Event | None = None,
    skip_event: threading.Event | None = None,
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

    Cancellation:
      - ``cancel_event`` short-circuits the entire suite (Stop button
        in the UI; SSE consumer disconnect).
      - ``skip_event`` short-circuits ONLY the current prompt,
        leaving the rest of the suite to run. The route layer flips
        this when the user clicks Skip on a row; we clear it after
        observing so the next prompt starts cleanly.
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
        skip_event=skip_event,
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
        skip_event=skip_event,
    )

    yield EvalFrame(done=True)


def _stream_phase(
    run_dict: dict[str, Any],
    role: str,
    cfg: EvalConfig,
    runs_root: Path,
    cancel_event: threading.Event | None,
    skip_event: threading.Event | None,
) -> Iterator[EvalFrame]:
    """Generate every prompt under one model configuration (base or
    adapter), tagging each frame with its prompt index + role.

    We translate the playground's GenerationToken stream into
    EvalFrames so the orchestrator's wire format stays uniform —
    the route's SSE serializer doesn't have to special-case base vs
    adapter or worry about per-prompt indexing.

    Skip handling: per-prompt cancel is implemented by passing a
    fresh ``threading.Event`` down to ``generate_stream`` for each
    prompt. When the suite-level ``skip_event`` is set we mirror it
    onto that per-prompt event, which the playground's HF
    StoppingCriteria observes — generation stops at the next forward
    pass and we move on. Clearing the suite-level skip flag at the
    boundary lets the next prompt run normally.
    """
    for idx, prompt in enumerate(cfg.prompts):
        if cancel_event is not None and cancel_event.is_set():
            return
        gcfg = playground.GenerationConfig(
            prompt=prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        # Combined per-prompt event: cancel sets it (terminates the
        # whole suite), skip sets it (terminates just this prompt).
        # Either way the playground stops the generator at the next
        # token boundary.
        per_prompt_cancel = threading.Event()

        def _watch_skip() -> None:
            if skip_event is None:
                return
            # Wait without burning a CPU. If skip fires during
            # generation, mirror to per_prompt_cancel; the watcher
            # exits when the prompt naturally finishes (we set the
            # cancel flag from inside the loop on completion).
            while not per_prompt_cancel.is_set():
                if skip_event.wait(timeout=0.05):
                    per_prompt_cancel.set()
                    skip_event.clear()
                    return

        if skip_event is not None:
            threading.Thread(target=_watch_skip, daemon=True).start()
        try:
            for tok in playground.generate_stream(
                run_dict, gcfg, runs_root, cancel_event=per_prompt_cancel,
            ):
                if tok.done:
                    # Per-prompt completion is implicit (next prompt
                    # starts streaming under the same role).
                    continue
                if tok.status is not None:
                    continue
                # Cancel-event observed mid-stream: stop forwarding
                # tokens and break out so we either advance to the
                # next prompt (skip) or exit (full cancel).
                if cancel_event is not None and cancel_event.is_set():
                    break
                yield EvalFrame(role=role, prompt_index=idx, text=tok.text)
        finally:
            # Wake the watcher so it exits even if skip never fired.
            per_prompt_cancel.set()


def compare_pairwise(
    left_run_dict: dict[str, Any],
    right_run_dict: dict[str, Any],
    cfg: EvalConfig,
    runs_root: Path,
    cancel_event: threading.Event | None = None,
    skip_event: threading.Event | None = None,
) -> Iterator[EvalFrame]:
    """Run the same prompts against two different run dicts.

    Generalises the base/adapter eval flow: instead of pairing a
    fixed base + adapter, the caller picks two arbitrary runs (or a
    base and a run, or two bases) and gets side-by-side outputs
    tagged with role ``"left"`` and ``"right"``. Useful for comparing
    two trained adapters against the same prompts — what F-A3 surfaces
    as the prompt comparator screen.

    Sequencing mirrors evaluate(): all left prompts, then all right
    prompts, so the playground's single-slot cache only swaps the
    model once. Cancel + skip handling is identical — the same
    ``_stream_phase`` is reused.
    """
    if not cfg.prompts:
        yield EvalFrame(done=True)
        return

    yield EvalFrame(status="Loading left model…")
    yield from _stream_phase(
        run_dict=left_run_dict,
        role="left",
        cfg=cfg,
        runs_root=runs_root,
        cancel_event=cancel_event,
        skip_event=skip_event,
    )

    if cancel_event is not None and cancel_event.is_set():
        yield EvalFrame(done=True)
        return

    yield EvalFrame(status="Loading right model…")
    yield from _stream_phase(
        run_dict=right_run_dict,
        role="right",
        cfg=cfg,
        runs_root=runs_root,
        cancel_event=cancel_event,
        skip_event=skip_event,
    )

    yield EvalFrame(done=True)


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
