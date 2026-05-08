import json
import threading
from pathlib import Path

from llm_chain_sidecar.inference import synth
from llm_chain_sidecar.inference.synth import (
    SynthConfig,
    _strip_fences,
    _try_parse_messages,
    base_run_dict,
    synthesize,
)


# --- _strip_fences ---------------------------------------------------


def test_strip_fences_removes_surrounding_markdown():
    text = '```json\n{"messages": []}\n```'
    assert _strip_fences(text) == '{"messages": []}'


def test_strip_fences_handles_unfenced_text():
    text = '{"messages": [1, 2]}'
    assert _strip_fences(text) == text


# --- _try_parse_messages ---------------------------------------------


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def test_parse_messages_accepts_well_formed_json():
    text = json.dumps(
        {"messages": [_msg("user", "hi"), _msg("assistant", "hello")]}
    )
    msgs = _try_parse_messages(text)
    assert msgs is not None
    assert len(msgs) == 2


def test_parse_messages_extracts_from_markdown_fenced_output():
    text = (
        "Sure! Here's the conversation:\n```json\n"
        + json.dumps(
            {"messages": [_msg("user", "q"), _msg("assistant", "a")]}
        )
        + "\n```"
    )
    msgs = _try_parse_messages(text)
    assert msgs is not None
    assert msgs[0]["content"] == "q"


def test_parse_messages_falls_back_to_first_balanced_object():
    """Models often dump prose around the JSON. The fallback should
    catch the first ``{...}`` block and parse it cleanly."""
    text = (
        "Here is what I generated:\n"
        + json.dumps(
            {"messages": [_msg("user", "x"), _msg("assistant", "y")]}
        )
        + "\nLet me know if you need adjustments."
    )
    msgs = _try_parse_messages(text)
    assert msgs is not None
    assert msgs[1]["role"] == "assistant"


def test_parse_messages_returns_none_for_garbage():
    assert _try_parse_messages("definitely not json") is None
    assert _try_parse_messages("") is None


def test_parse_messages_rejects_single_message_rows():
    """A 'conversation' with only one turn isn't useful training data —
    treat it as a parse failure so the user can retry that row."""
    text = json.dumps({"messages": [_msg("user", "lonely")]})
    assert _try_parse_messages(text) is None


def test_parse_messages_rejects_missing_role_or_content():
    text = json.dumps({"messages": [{"role": "user"}, _msg("assistant", "x")]})
    assert _try_parse_messages(text) is None


# --- base_run_dict ---------------------------------------------------


def test_base_run_dict_uses_stable_synth_id():
    rd = base_run_dict("acme/cool-model", "mlx")
    assert rd["output_dir"] == ""
    assert rd["config"]["model_id"] == "acme/cool-model"
    assert rd["config"]["backend"] == "mlx"
    # Same args → same id, so the playground's per-run cache key is stable
    # across calls and the model stays warm between rows.
    assert rd["id"] == base_run_dict("acme/cool-model", "mlx")["id"]


# --- synthesize (with the playground stream stubbed out) -------------


def test_synthesize_yields_one_row_per_count_with_stub(monkeypatch):
    """End-to-end shape test: the orchestrator should produce
    ``count`` row frames terminated by a single done frame, regardless
    of which playground backend is wired underneath. We monkeypatch
    the underlying _collect_stream so the test stays unit-fast."""
    counter = {"n": 0}

    def _fake_collect(run_dict, gcfg, runs_root, cancel_event):
        idx = counter["n"]
        counter["n"] += 1
        return json.dumps(
            {
                "messages": [
                    _msg("user", f"q{idx}"),
                    _msg("assistant", f"a{idx}"),
                ]
            }
        )

    monkeypatch.setattr(synth, "_collect_stream", _fake_collect)

    cfg = SynthConfig(topic="t", style="s", count=3, max_tokens=64)
    frames = list(
        synthesize({"id": "r", "config": {}, "output_dir": ""}, cfg, Path("."))
    )
    rows = [f for f in frames if f.index >= 0]
    done = [f for f in frames if f.done]
    assert len(rows) == 3
    assert all(f.parsed for f in rows)
    assert len(done) == 1
    assert done[0].stats == {"total": 3, "parsed_ok": 3, "parse_failed": 0}


def test_synthesize_retries_once_on_parse_failure(monkeypatch):
    """A single bad first attempt should be followed by a retry that
    counts toward parsed_ok when it succeeds. This exercises the
    "stricter prompt" fallback the synthesiser does inline."""
    call_count = {"n": 0}

    def _fake_collect(run_dict, gcfg, runs_root, cancel_event):
        call_count["n"] += 1
        # First call: garbage. Subsequent calls: well-formed JSON.
        if call_count["n"] == 1:
            return "totally not parseable"
        return json.dumps(
            {"messages": [_msg("user", "q"), _msg("assistant", "a")]}
        )

    monkeypatch.setattr(synth, "_collect_stream", _fake_collect)

    frames = list(
        synthesize(
            {"id": "r", "config": {}, "output_dir": ""},
            SynthConfig(topic="t", style="s", count=1),
            Path("."),
        )
    )
    rows = [f for f in frames if f.index >= 0]
    assert len(rows) == 1
    assert rows[0].parsed is True
    assert call_count["n"] == 2  # primary + one retry


def test_synthesize_emits_unparsed_row_after_two_failed_attempts(monkeypatch):
    """Both attempts fail → row frame with parsed=False and the raw
    text so the UI can show the user what came back."""

    def _fake_collect(run_dict, gcfg, runs_root, cancel_event):
        return "still not json"

    monkeypatch.setattr(synth, "_collect_stream", _fake_collect)

    frames = list(
        synthesize(
            {"id": "r", "config": {}, "output_dir": ""},
            SynthConfig(topic="t", style="s", count=1),
            Path("."),
        )
    )
    rows = [f for f in frames if f.index >= 0]
    done = [f for f in frames if f.done]
    assert len(rows) == 1
    assert rows[0].parsed is False
    assert rows[0].raw_text == "still not json"
    assert done[0].stats["parse_failed"] == 1


def test_synthesize_short_circuits_on_cancel(monkeypatch):
    """Setting the cancel event before the first row should stop the
    loop and still terminate cleanly with a done frame."""

    def _fake_collect(run_dict, gcfg, runs_root, cancel_event):
        return json.dumps(
            {"messages": [_msg("user", "q"), _msg("assistant", "a")]}
        )

    monkeypatch.setattr(synth, "_collect_stream", _fake_collect)

    cancel = threading.Event()
    cancel.set()
    frames = list(
        synthesize(
            {"id": "r", "config": {}, "output_dir": ""},
            SynthConfig(topic="t", style="s", count=10),
            Path("."),
            cancel_event=cancel,
        )
    )
    # No row frames; done with zero counts.
    assert all(f.index == -1 for f in frames)
    assert frames[-1].done
    assert frames[-1].stats["total"] == 0
