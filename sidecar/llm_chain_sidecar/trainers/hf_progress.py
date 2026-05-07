"""Capture huggingface_hub download progress as queue events.

Both huggingface_hub and transformers route their download bars through
``tqdm.auto.tqdm``. The library never exposed a per-download callback, so we
patch ``tqdm.auto.tqdm.display`` for the duration of a context and emit a
structured event each time a download bar advances. On exit we restore the
original method so unrelated callers — tests, scripts — see vanilla tqdm.
"""

import queue
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def emit_hf_download_progress(events: "queue.Queue[dict | None]") -> Iterator[None]:
    import tqdm.auto

    original_display = tqdm.auto.tqdm.display
    last_emit_pct: dict[int, float] = {}

    def patched_display(self, msg=None, pos=None):
        total = getattr(self, "total", None)
        n = getattr(self, "n", 0) or 0
        if total and total > 0:
            pct = (n / total) * 100
            key = id(self)
            prev = last_emit_pct.get(key, -1.0)
            # Throttle: at most one event per 1% advance, plus the final tick.
            # Without this an HF download can fire thousands of events for a
            # single file and saturate the SSE channel.
            if pct - prev >= 1.0 or n >= total:
                last_emit_pct[key] = pct
                try:
                    events.put_nowait({
                        "type": "download",
                        "bytes_done": int(n),
                        "bytes_total": int(total),
                        "desc": getattr(self, "desc", "") or "",
                    })
                except queue.Full:
                    pass
        return original_display(self, msg, pos)

    tqdm.auto.tqdm.display = patched_display
    try:
        yield
    finally:
        tqdm.auto.tqdm.display = original_display
