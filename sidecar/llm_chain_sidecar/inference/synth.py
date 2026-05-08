"""Synthetic data generator: turn an existing chat-capable run (or a
base model) into a stream of (user, assistant) pairs the user can save
as a JSONL training set.

We lean on the playground's per-backend ``generate_stream`` so the
heavy lifting (model load, cache reuse, cancel handling) stays in one
place. This module only orchestrates: build a generator prompt, buffer
each row's output, parse + validate the JSON shape, emit one frame per
attempt with parse status, retry once on parse failure.

The output is intentionally not written to disk here. The route emits
SSE frames; the frontend collects rows in memory and passes them
through the existing /api/datasets/build endpoint when the user
chooses to save. Keeping write-out separate means the user can
discard a bad batch without leaving turds in ~/.llm-chain/datasets/.
"""
from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import playground


@dataclass(frozen=True)
class SynthConfig:
    """Inputs the route forwards to the orchestrator.

    ``count`` is bounded to 100 at the route layer — a single SSE
    stream of more rows would consume gigabytes of model output and
    keep the user staring at a progress bar for hours.
    """

    topic: str
    style: str
    count: int = 10
    max_tokens: int = 512
    temperature: float = 0.9  # higher than eval — synth wants variety,
                              # not deterministic-looking duplicates
    seed_prompts: tuple[str, ...] = ()


@dataclass(frozen=True)
class SynthFrame:
    """One streamed update from the synthesiser.

    Three event types share the shape:
      - ``status``: phase transitions ("Loading model…", "Generating
        row 5/10…"). Not appended to any visible buffer.
      - ``row``: ``index`` is the 0-based row position, ``messages``
        the parsed conversation (None when parse failed), ``raw_text``
        the model's full output for that attempt (always set, so the
        UI can show the raw output even when parsing failed).
      - ``done``: terminal frame with ``stats`` dict carrying counts
        of total / parsed_ok / parse_failed.
    """

    status: str | None = None
    index: int = -1
    messages: list[dict] | None = None
    raw_text: str | None = None
    parsed: bool = False
    done: bool = False
    stats: dict | None = None


# Markers we strip from the model's output before JSON parsing. Many
# models prefix or wrap their answer in ``json``-fenced markdown when
# we ask for JSON, so we tolerate both fenced and bare output.
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def _system_prompt(topic: str, style: str) -> str:
    """The prompt that nudges the model into producing one parseable
    conversation per call.

    Kept short and concrete — long meta-instructions tend to drift
    into "Sure, I'd be happy to…" framing instead of the actual
    output. The example schema is included so the model can copy the
    keys verbatim instead of inventing variants.
    """
    return (
        "You are generating one training conversation in JSON. "
        "Produce ONLY a JSON object with this exact schema:\n"
        '{"messages":[{"role":"user","content":"..."},'
        '{"role":"assistant","content":"..."}]}\n'
        f"Topic: {topic}\n"
        f"Style: {style}\n"
        "Make the conversation concrete and useful. Do not wrap the "
        "JSON in markdown fences. Do not add commentary before or "
        "after."
    )


def _user_prompt_for_index(idx: int, total: int, seed_prompts: tuple[str, ...]) -> str:
    """The per-row instruction.

    Without a varying prompt, the model returns near-identical rows
    even at temperature 0.9 — the seed_prompts list (when present)
    rotates topics across rows, falling back to a generic varied-
    examples prompt when the user didn't seed any.
    """
    if seed_prompts:
        seed = seed_prompts[idx % len(seed_prompts)]
        return (
            f"Generate conversation {idx + 1} of {total}. "
            f"Make this one focus on: {seed}"
        )
    return (
        f"Generate conversation {idx + 1} of {total}. "
        "Vary the user's question or task from any previous output."
    )


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _try_parse_messages(text: str) -> list[dict] | None:
    """Best-effort JSON parse with multi-object fallback.

    1. Strip markdown fences and try the whole thing.
    2. Walk every top-level balanced ``{...}`` block and try each.

    Returns the messages list when parsing produced a well-shaped
    chat row; None otherwise. The caller surfaces the failure to the
    UI so the user can see the raw output and decide whether to keep
    the batch.

    Why all objects, not just the first: thinking-mode models often
    emit ``{"reasoning":"..."} {"messages":[...]}`` — the first object
    has no ``messages`` field and was previously the only candidate
    tried, so the row registered as a parse failure even though the
    intended payload was present.
    """
    if not text or not text.strip():
        return None

    candidates: list[str] = [_strip_fences(text)]
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : i + 1])
                start = -1

    for body in candidates:
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            continue
        msgs = obj.get("messages") if isinstance(obj, dict) else None
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue
        ok = True
        for m in msgs:
            if not isinstance(m, dict):
                ok = False
                break
            if "role" not in m or "content" not in m:
                ok = False
                break
            if not isinstance(m["content"], str):
                ok = False
                break
        if ok:
            return msgs
    return None


def base_run_dict(model_id: str, backend: str) -> dict[str, Any]:
    """Synthesise a run_dict the playground loader treats as base-only.

    Mirrors :func:`eval_suite._base_only_run_dict` — the playground
    cache keys on ``run_id`` so we use a stable distinct id and an
    empty ``output_dir`` to signal "skip the adapter step". Useful
    when the user wants to seed a synthetic dataset from a fresh
    model in the registry rather than from an existing adapter.
    """
    return {
        "id": f"synth-base:{model_id}:{backend}",
        "config": {"model_id": model_id, "backend": backend},
        "output_dir": "",
    }


def synthesize(
    run_dict: dict[str, Any],
    cfg: SynthConfig,
    runs_root: Path,
    cancel_event: threading.Event | None = None,
) -> Iterator[SynthFrame]:
    """Generate ``cfg.count`` conversation rows one at a time.

    For each row we build a conversation prompt, run the playground's
    streaming generator to collect the entire output, parse it, and
    emit one ``row`` frame. On parse failure we retry once with a
    sterner system prompt; if that also fails we yield the row with
    ``parsed=False`` and let the user decide.

    Cancellation: the caller's ``cancel_event`` short-circuits the
    whole loop. We propagate it down to ``generate_stream`` per row
    so an in-flight generate also stops at the next token boundary.
    """
    if cfg.count <= 0:
        yield SynthFrame(
            done=True,
            stats={"total": 0, "parsed_ok": 0, "parse_failed": 0},
        )
        return

    sys_prompt = _system_prompt(cfg.topic, cfg.style)
    parsed_ok = 0
    parse_failed = 0

    yield SynthFrame(status=f"Preparing to generate {cfg.count} rows…")

    for i in range(cfg.count):
        if cancel_event is not None and cancel_event.is_set():
            break

        yield SynthFrame(status=f"Generating row {i + 1}/{cfg.count}…")

        user_prompt = _user_prompt_for_index(i, cfg.count, cfg.seed_prompts)
        full_prompt = sys_prompt + "\n\n" + user_prompt
        gcfg = playground.GenerationConfig(
            prompt=full_prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )

        text = _collect_stream(run_dict, gcfg, runs_root, cancel_event)
        messages = _try_parse_messages(text)

        if messages is None:
            # One retry with a stricter prompt — many models drift the
            # second time around when explicitly told to ONLY emit JSON.
            strict_prompt = (
                full_prompt
                + "\n\nIMPORTANT: Reply with ONLY the JSON object — no prose, "
                + "no markdown fences. The JSON must parse on the first try."
            )
            gcfg = playground.GenerationConfig(
                prompt=strict_prompt,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            retry_text = _collect_stream(run_dict, gcfg, runs_root, cancel_event)
            messages = _try_parse_messages(retry_text)
            if messages is None:
                parse_failed += 1
                yield SynthFrame(
                    index=i,
                    raw_text=retry_text or text,
                    parsed=False,
                )
                continue
            text = retry_text

        parsed_ok += 1
        yield SynthFrame(
            index=i,
            messages=messages,
            raw_text=text,
            parsed=True,
        )

    yield SynthFrame(
        done=True,
        stats={
            "total": parsed_ok + parse_failed,
            "parsed_ok": parsed_ok,
            "parse_failed": parse_failed,
        },
    )


def _collect_stream(
    run_dict: dict[str, Any],
    gcfg: playground.GenerationConfig,
    runs_root: Path,
    cancel_event: threading.Event | None,
) -> str:
    """Drain ``generate_stream`` for one prompt and return the full text.

    Status frames are dropped — they're the playground's
    "Loading model…" hints which the synth orchestrator surfaces at
    its own granularity. Cancel propagates: a per-row event mirrors
    the suite-level cancel so closing the SSE consumer aborts the
    in-flight generate at the next token boundary.
    """
    per_row = threading.Event()

    def _watcher() -> None:
        if cancel_event is None:
            return
        while not per_row.is_set():
            if cancel_event.wait(timeout=0.05):
                per_row.set()
                return

    if cancel_event is not None:
        threading.Thread(target=_watcher, daemon=True).start()

    parts: list[str] = []
    try:
        for tok in playground.generate_stream(
            run_dict, gcfg, runs_root, cancel_event=per_row,
        ):
            if tok.done:
                break
            if tok.status is not None:
                continue
            parts.append(tok.text)
    finally:
        per_row.set()
    return "".join(parts)
