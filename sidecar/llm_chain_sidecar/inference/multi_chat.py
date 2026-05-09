"""Multi-adapter chat (F-B8).

Loads N adapters that share a base model and streams their responses
to the same user turn side-by-side. The playground's N-slot LRU
holds them warm across turns so a follow-up question only re-uses
already-loaded weights.

Per-adapter histories: each adapter sees its own conversation track
that diverges after turn 1. The route accepts a list of (run_id,
messages) pairs from the client; we generate sequentially per
adapter and tag each token frame with the originating adapter id.
Sequential rather than parallel because mlx_lm + transformers don't
parallelise gracefully under a single Python process and the cache
read patterns are simpler this way.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import playground


@dataclass(frozen=True)
class MultiChatTurn:
    """One adapter's slice of a multi-adapter chat turn.

    ``messages`` is the full per-adapter history including the new
    user prompt. The orchestrator hands the messages straight to
    ``GenerationConfig.messages``; the playground builds the chat-
    template prompt internally.
    """

    run_dict: dict[str, Any]
    messages: tuple[dict, ...]


@dataclass(frozen=True)
class MultiChatFrame:
    """Wire frame for the SSE stream.

    Three event types share the shape:
      - ``status``: phase transitions ("Generating with adapter A…").
      - ``token``: ``adapter_id`` + ``text`` delta to append.
      - ``done``: ``done=True`` after every adapter finishes its turn.
    """

    adapter_id: str = ""
    text: str = ""
    status: str | None = None
    done: bool = False


def stream_turn(
    turns: list[MultiChatTurn],
    *,
    runs_root: Path,
    max_tokens: int = 256,
    temperature: float = 0.7,
    cancel_event: threading.Event | None = None,
) -> Iterator[MultiChatFrame]:
    """Iterate per-adapter, stream tokens for each, terminate with done.

    Cancel: a single ``cancel_event`` aborts the whole turn — the
    in-flight adapter stops at the next token boundary and the
    remaining adapters are skipped. Per-adapter cancel would let the
    user kill one column without nuking the others, but the UI doesn't
    expose that affordance yet; v1 is "stop the whole turn".
    """
    if not turns:
        yield MultiChatFrame(done=True)
        return

    for turn in turns:
        if cancel_event is not None and cancel_event.is_set():
            break
        adapter_id = turn.run_dict["id"]
        yield MultiChatFrame(
            adapter_id=adapter_id,
            status=f"Generating with adapter {adapter_id}…",
        )
        # One per-adapter cancel event mirrors the suite-level one so
        # the playground's StoppingCriteria can short-circuit at the
        # next token. Without this the worker would run to max_tokens.
        per_adapter_cancel = threading.Event()

        def _watch_outer() -> None:
            if cancel_event is None:
                return
            while not per_adapter_cancel.is_set():
                if cancel_event.wait(timeout=0.05):
                    per_adapter_cancel.set()
                    return

        if cancel_event is not None:
            threading.Thread(target=_watch_outer, daemon=True).start()

        gcfg = playground.GenerationConfig(
            messages=turn.messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            for tok in playground.generate_stream(
                turn.run_dict, gcfg, runs_root, cancel_event=per_adapter_cancel,
            ):
                if tok.done:
                    break
                if tok.status is not None:
                    # Surface the playground's own status frames under
                    # this adapter's tag so the UI can disambiguate
                    # "loading model A" from "loading model B".
                    yield MultiChatFrame(
                        adapter_id=adapter_id, status=tok.status,
                    )
                    continue
                if cancel_event is not None and cancel_event.is_set():
                    break
                yield MultiChatFrame(adapter_id=adapter_id, text=tok.text)
        finally:
            per_adapter_cancel.set()

    yield MultiChatFrame(done=True)
