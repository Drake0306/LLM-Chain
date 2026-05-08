"""Unit tests for the inference playground module.

Covers the orchestration layer (cache hits / misses, status emission,
cancel propagation) without loading a real model. Heavy backends
(``mlx_lm``, ``transformers``) are imported lazily inside
``_load_for_run``, so as long as we don't reach that path the tests
run on any host.

A handful of tests reach into the MLX backend's stream layer and
``patch("mlx_lm.stream_generate", ...)`` — that resolves the
``mlx_lm`` module at decorator-entry time, so it crashes with
ModuleNotFoundError on hosts without the macOS extra (e.g. CI's
generic macos-14 runner). Those tests carry an explicit
``mlx_lm_required`` skip so the rest of the file still runs there.
"""
import importlib.util
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_chain_sidecar.inference import playground

mlx_lm_required = pytest.mark.skipif(
    importlib.util.find_spec("mlx_lm") is None,
    reason="mlx_lm not installed on this host",
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Each test starts with an empty inference cache so cross-test
    state doesn't bleed in (e.g. a test that primed the cache making
    the next test's "cache miss" path skip the load)."""
    playground.free_cache()
    yield
    playground.free_cache()


def _stub_entry(run_id: str, backend: str = "cuda") -> playground._CacheEntry:
    """Build a minimal cache entry without invoking real backend
    loaders. The downstream stream functions are patched per-test, so
    the obj contents don't have to be realistic."""
    return playground._CacheEntry(run_id=run_id, backend=backend, obj={})


def test_generate_stream_emits_loading_status_on_cold_cache(monkeypatch):
    """First playground call against an empty cache should produce a
    'Loading model…' status frame so the UI can replace its spinner
    with informative text."""
    run_dict = {"id": "abc123", "config": {"backend": "cuda"}, "output_dir": "/tmp/x"}

    def fake_load(rd, _root):
        return _stub_entry(run_id=rd["id"], backend="cuda")

    monkeypatch.setattr(playground, "_load_for_run", fake_load)
    monkeypatch.setattr(
        playground, "_stream_hf",
        lambda entry, cfg, cancel_event=None: iter(
            [playground.GenerationToken(text="hi"), playground.GenerationToken(done=True)]
        ),
    )

    cfg = playground.GenerationConfig(prompt="hi")
    out = list(playground.generate_stream(run_dict, cfg, Path("/tmp")))
    statuses = [t.status for t in out if t.status]
    assert statuses, "expected at least one status frame on cold cache"
    assert "Loading" in statuses[0]


def test_generate_stream_emits_switching_status_on_different_run(monkeypatch):
    """When the cached model belongs to a different run, the UI
    needs to know we're paying the swap cost — not just a new model
    load. The status string names both run ids so the user can map
    it back to whichever tab they came from."""
    monkeypatch.setattr(
        playground, "_load_for_run",
        lambda rd, _root: _stub_entry(rd["id"]),
    )
    monkeypatch.setattr(
        playground, "_stream_hf",
        lambda entry, cfg, cancel_event=None: iter([playground.GenerationToken(done=True)]),
    )

    # Prime the cache with run "old".
    list(
        playground.generate_stream(
            {"id": "old", "config": {"backend": "cuda"}, "output_dir": "/tmp/x"},
            playground.GenerationConfig(prompt="x"),
            Path("/tmp"),
        )
    )
    # Now ask for run "new" — should emit a "Switching cached model…" status.
    out = list(
        playground.generate_stream(
            {"id": "new", "config": {"backend": "cuda"}, "output_dir": "/tmp/y"},
            playground.GenerationConfig(prompt="x"),
            Path("/tmp"),
        )
    )
    statuses = [t.status for t in out if t.status]
    assert any("Switching" in s and "old" in s and "new" in s for s in statuses)


def test_generate_stream_skips_status_on_warm_cache(monkeypatch):
    """A repeated prompt against the same run shouldn't repeat the
    'Loading model…' status — the user knows the model is loaded
    after the first call."""
    monkeypatch.setattr(
        playground, "_load_for_run",
        lambda rd, _root: _stub_entry(rd["id"]),
    )
    monkeypatch.setattr(
        playground, "_stream_hf",
        lambda entry, cfg, cancel_event=None: iter(
            [playground.GenerationToken(text="hi"), playground.GenerationToken(done=True)]
        ),
    )

    # Prime.
    list(
        playground.generate_stream(
            {"id": "abc", "config": {"backend": "cuda"}, "output_dir": "/tmp/x"},
            playground.GenerationConfig(prompt="x"),
            Path("/tmp"),
        )
    )
    # Same run — no status frame this time.
    second = list(
        playground.generate_stream(
            {"id": "abc", "config": {"backend": "cuda"}, "output_dir": "/tmp/x"},
            playground.GenerationConfig(prompt="x"),
            Path("/tmp"),
        )
    )
    assert all(t.status is None for t in second)


def test_generate_stream_passes_cancel_event_to_backend(monkeypatch):
    """The route hands a cancel_event into generate_stream; that has
    to make it all the way down to the per-backend streamer. Without
    this wiring the HF StoppingCriteria has nothing to poll."""
    monkeypatch.setattr(
        playground, "_load_for_run",
        lambda rd, _root: _stub_entry(rd["id"]),
    )
    captured: dict[str, object] = {}

    def fake_stream_hf(entry, cfg, cancel_event=None):
        captured["got"] = cancel_event
        yield playground.GenerationToken(done=True)

    monkeypatch.setattr(playground, "_stream_hf", fake_stream_hf)
    ev = threading.Event()
    list(
        playground.generate_stream(
            {"id": "abc", "config": {"backend": "cuda"}, "output_dir": "/tmp/x"},
            playground.GenerationConfig(prompt="x"),
            Path("/tmp"),
            cancel_event=ev,
        )
    )
    assert captured["got"] is ev


@mlx_lm_required
def test_mlx_stream_skips_empty_text_deltas():
    """mlx_lm yields GenerationResponse objects whose ``.text`` is
    LEGITIMATELY empty mid-generation (BPE token whose bytes haven't
    accumulated into a full UTF-8 sequence yet). The bug this pins
    against: older code did ``getattr(r, 'text', None) or str(r)``,
    which turned every empty delta into a full ``GenerationResponse(
    text='', token=216, ...)`` repr in the user's chat output."""
    entry = playground._CacheEntry(
        run_id="r",
        backend="mlx",
        obj={"model": object(), "tokenizer": _StubTokenizer()},
    )
    # Two empty deltas (typical mid-token byte accumulation), then a
    # real word, then EOS-equivalent empty + finish.
    fake_chunks = [
        _FakeMlxResponse(""),
        _FakeMlxResponse(""),
        _FakeMlxResponse("Hello"),
        _FakeMlxResponse(""),
    ]

    def fake_stream_generate(model, tok, prompt, max_tokens):
        yield from fake_chunks

    with patch("mlx_lm.stream_generate", fake_stream_generate, create=True):
        cfg = playground.GenerationConfig(prompt="hi")
        out = list(playground._stream_mlx(entry, cfg, cancel_event=None))

    text_tokens = [t for t in out if not t.done]
    # Only the one non-empty delta gets emitted. No GenerationResponse
    # repr leaks into the stream.
    assert len(text_tokens) == 1
    assert text_tokens[0].text == "Hello"
    # Importantly: the emitted text is "Hello", not a stringified
    # GenerationResponse — make that explicit so a regression to the
    # old fallback fails this assertion.
    assert "GenerationResponse" not in text_tokens[0].text


@mlx_lm_required
def test_mlx_stream_breaks_on_cancel_event_set():
    """mlx's stream_generate is itself a generator — abandoning the
    iteration usually ends the model loop, but we also short-circuit
    one step earlier when cancel is already armed so callers don't
    pay an extra forward pass after Stop."""
    entry = playground._CacheEntry(
        run_id="r",
        backend="mlx",
        obj={"model": object(), "tokenizer": _StubTokenizer()},
    )
    fake_chunks = [_FakeMlxResponse("a"), _FakeMlxResponse("b"), _FakeMlxResponse("c")]
    cancel = threading.Event()

    def fake_stream_generate(model, tok, prompt, max_tokens):
        for c in fake_chunks:
            if cancel.is_set():
                return
            yield c

    with patch.dict("sys.modules"), patch(
        "mlx_lm.stream_generate", fake_stream_generate, create=True
    ):
        # Simulate a Stop click before any token arrived.
        cancel.set()
        cfg = playground.GenerationConfig(prompt="x")
        out = list(playground._stream_mlx(entry, cfg, cancel_event=cancel))
    # Only the trailing done=True frame; no token was emitted because
    # cancel fired before the first yield.
    text_tokens = [t for t in out if not t.done]
    assert text_tokens == []
    assert out[-1].done is True


class _StubTokenizer:
    """Minimal tokenizer stand-in. _build_chat_prompt only reads
    chat_template; everything else is bypassed when we mock the
    stream_generate function above."""
    chat_template = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return ""


class _FakeMlxResponse:
    def __init__(self, text):
        self.text = text
