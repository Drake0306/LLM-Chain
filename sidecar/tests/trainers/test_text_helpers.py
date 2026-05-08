"""Unit tests for the shared chat/text conversion helpers.

These exercise the pieces that the HF and CPU trainers used to inline.
Catching regressions here means we don't have to fire up the real HF
pipeline (slow + needs network) to verify a chat row gets templated
correctly or that pad_token gets a sensible fallback.
"""
import queue
import time

import pytest

from llm_chain_sidecar.datasets.types import DatasetFormat
from llm_chain_sidecar.trainers._text import (
    ensure_pad_token,
    row_to_text,
    run_in_background_with_sentinel,
)


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer. Tracks pad/eos/unk + chat_template
    behaviour. Real HF tokenizers do far more, but this captures every field
    the helpers actually read."""

    def __init__(
        self,
        *,
        pad_token=None,
        eos_token=None,
        unk_token=None,
        chat_template=None,
    ) -> None:
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.chat_template = chat_template
        self.added: dict[str, str] = {}

    def add_special_tokens(self, mapping):
        self.added.update(mapping)
        for k, v in mapping.items():
            setattr(self, k, v)

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        # Stand-in for the HF templater: produce a deterministic, role-aware
        # rendering that's easy to assert on without depending on a specific
        # model's template string.
        assert tokenize is False
        return "|".join(f"{m['role']}={m['content']}" for m in msgs)


def test_ensure_pad_token_no_op_when_already_set_returns_false():
    tok = _FakeTokenizer(pad_token="<pad>")
    grew = ensure_pad_token(tok)
    assert tok.pad_token == "<pad>"
    assert tok.added == {}
    assert grew is False


def test_ensure_pad_token_eos_fallback_returns_false():
    """Reusing an existing token doesn't grow the vocabulary, so the
    trainer can skip resize_token_embeddings."""
    tok = _FakeTokenizer(eos_token="</s>")
    grew = ensure_pad_token(tok)
    assert tok.pad_token == "</s>"
    assert grew is False


def test_ensure_pad_token_unk_fallback_returns_false():
    tok = _FakeTokenizer(unk_token="<unk>")
    grew = ensure_pad_token(tok)
    assert tok.pad_token == "<unk>"
    assert grew is False


def test_ensure_pad_token_literal_pad_branch_returns_true():
    """The last-resort branch adds a brand new token to the vocabulary,
    which means the trainer MUST call model.resize_token_embeddings or
    the forward pass crashes on an out-of-bounds id. The return flag is
    how the trainer knows to do that."""
    tok = _FakeTokenizer()
    grew = ensure_pad_token(tok)
    assert tok.pad_token == "[PAD]"
    assert grew is True


def test_row_to_text_chat_uses_apply_chat_template():
    """The previous to_text inlined `f"{role}: {content}"` which bypassed
    the tokenizer's chat template entirely — even on chat-tuned models —
    so fine-tunes were silently degraded. Verify we go through
    apply_chat_template now."""
    tok = _FakeTokenizer(chat_template="{% for m in messages %}…{% endfor %}")
    row = {"messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]}
    text = row_to_text(row, DatasetFormat.JSONL_CHAT, tok, text_column=None)
    assert text == "user=hello|assistant=hi"


def test_row_to_text_chat_raises_when_template_missing():
    """Defense in depth: if a model with no chat_template somehow gets
    past the API-level chat_capable validator (e.g. unknown HF id), the
    trainer raises a clear actionable error instead of letting the HF
    layer surface its 'Cannot use chat template functions' traceback."""
    tok = _FakeTokenizer(chat_template=None)
    row = {"messages": [{"role": "user", "content": "x"}]}
    with pytest.raises(ValueError, match="chat_template"):
        row_to_text(row, DatasetFormat.JSONL_CHAT, tok, text_column=None)


def test_row_to_text_vision_format_rejected_in_text_trainer():
    tok = _FakeTokenizer(chat_template="present")
    row = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    with pytest.raises(ValueError, match="vision"):
        row_to_text(row, DatasetFormat.JSONL_CHAT_VISION, tok, text_column=None)


def test_row_to_text_csv_uses_text_column():
    tok = _FakeTokenizer()
    row = {"id": 1, "body": "hello"}
    assert row_to_text(row, DatasetFormat.CSV, tok, text_column="body") == "hello"


def test_row_to_text_csv_raises_with_helpful_message_when_column_missing():
    tok = _FakeTokenizer()
    row = {"id": 1, "body": "hello"}
    with pytest.raises(ValueError, match="missing"):
        row_to_text(row, DatasetFormat.CSV, tok, text_column="missing")


def test_row_to_text_text_dir_uses_text_field():
    tok = _FakeTokenizer()
    row = {"text": "alpha"}
    assert row_to_text(row, DatasetFormat.TEXT_DIR, tok, text_column=None) == "alpha"


def test_row_to_text_hf_hub_falls_back_through_common_columns():
    tok = _FakeTokenizer()
    row = {"content": "from-content"}
    assert row_to_text(row, DatasetFormat.HF_HUB, tok, text_column=None) == "from-content"


def test_row_to_text_hf_hub_prefers_user_specified_column():
    tok = _FakeTokenizer()
    row = {"text": "default", "body": "specified"}
    assert (
        row_to_text(row, DatasetFormat.HF_HUB, tok, text_column="body")
        == "specified"
    )


def test_row_to_text_hf_hub_raises_when_no_text_field():
    tok = _FakeTokenizer()
    row = {"id": 1, "label": "spam"}
    with pytest.raises(ValueError, match="text column"):
        row_to_text(row, DatasetFormat.HF_HUB, tok, text_column=None)


def test_run_in_background_with_sentinel_clean_exit_pushes_none():
    events: queue.Queue = queue.Queue()
    events.put({"step": 1, "loss": 1.0})

    def worker() -> None:
        events.put({"step": 2, "loss": 0.9})
        # mimic on_train_end pushing None on success
        # (the helper guarantees this even if the worker forgets, but a real
        # HF Trainer will push it itself).

    t = run_in_background_with_sentinel(worker, events)
    t.join(timeout=2)
    assert not t.is_alive()
    drained = []
    while True:
        try:
            drained.append(events.get_nowait())
        except queue.Empty:
            break
    # Two real events plus the helper's None sentinel.
    assert drained[-1] is None
    assert {"step": 1, "loss": 1.0} in drained
    assert {"step": 2, "loss": 0.9} in drained


def test_run_in_background_with_sentinel_propagates_exception_through_queue():
    """Pre-fix: if hf.train() raised before on_train_end fired, the
    sentinel was never pushed and the consumer's events.get() blocked
    forever. This contract test pins the new behaviour: the helper
    guarantees an error event AND a None sentinel reach the queue.
    """
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        raise RuntimeError("CUDA out of memory")

    t = run_in_background_with_sentinel(worker, events)
    t.join(timeout=2)
    assert not t.is_alive()

    err = events.get(timeout=1)
    assert err.get("type") == "error"
    assert isinstance(err["exception"], RuntimeError)
    assert "CUDA out of memory" in str(err["exception"])

    # Sentinel reaches consumer right after the error event so
    # `while True: events.get()` always terminates.
    assert events.get(timeout=1) is None


def test_run_in_background_with_sentinel_no_consumer_hang_under_load():
    """End-to-end: the canonical consumer pattern (loop until None) must
    terminate quickly even when the worker raises immediately, with no
    timeout-based mitigation."""
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        raise ValueError("dataset row 5 missing 'messages'")

    run_in_background_with_sentinel(worker, events)
    start = time.monotonic()
    drained = []
    while True:
        ev = events.get(timeout=2)
        if ev is None:
            break
        drained.append(ev)
    # If the deadlock guard is missing this test times out at events.get(2).
    assert time.monotonic() - start < 2
    assert len(drained) == 1
    assert drained[0]["type"] == "error"
