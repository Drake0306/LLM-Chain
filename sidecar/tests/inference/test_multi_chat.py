import threading
from pathlib import Path

from llm_chain_sidecar.inference import multi_chat, playground


def _stub_run(run_id: str) -> dict:
    return {"id": run_id, "config": {"backend": "cuda"}, "output_dir": "/tmp/x"}


def _fake_generate(run_dict, gcfg, runs_root, cancel_event=None):
    """Yield two text deltas + done. Tags each delta with the run_id
    via the run_dict so the test can verify routing without hitting
    the real model loaders."""
    rid = run_dict["id"]
    yield playground.GenerationToken(text=f"hello-from-{rid} ")
    yield playground.GenerationToken(text=f"part2-{rid}")
    yield playground.GenerationToken(done=True)


def test_stream_turn_yields_one_segment_per_adapter(monkeypatch):
    monkeypatch.setattr(playground, "generate_stream", _fake_generate)
    turns = [
        multi_chat.MultiChatTurn(
            run_dict=_stub_run("a"),
            messages=({"role": "user", "content": "hi"},),
        ),
        multi_chat.MultiChatTurn(
            run_dict=_stub_run("b"),
            messages=({"role": "user", "content": "hi"},),
        ),
    ]
    frames = list(multi_chat.stream_turn(turns, runs_root=Path("/tmp")))
    a_text = "".join(
        f.text for f in frames if f.adapter_id == "a" and f.text
    )
    b_text = "".join(
        f.text for f in frames if f.adapter_id == "b" and f.text
    )
    assert "from-a" in a_text
    assert "from-b" in b_text
    assert frames[-1].done is True


def test_stream_turn_emits_per_adapter_status(monkeypatch):
    monkeypatch.setattr(playground, "generate_stream", _fake_generate)
    turns = [
        multi_chat.MultiChatTurn(
            run_dict=_stub_run("alpha"),
            messages=({"role": "user", "content": "hi"},),
        ),
    ]
    frames = list(multi_chat.stream_turn(turns, runs_root=Path("/tmp")))
    statuses = [f for f in frames if f.status is not None]
    assert any(
        f.adapter_id == "alpha" and "alpha" in (f.status or "")
        for f in statuses
    )


def test_stream_turn_short_circuits_on_cancel(monkeypatch):
    """A pre-set cancel event should skip every adapter and still
    terminate cleanly with a done frame. The whole-turn cancel is the
    UI's "Stop" button; per-adapter cancel isn't yet exposed."""

    def _slow_generate(run_dict, gcfg, runs_root, cancel_event=None):
        # Pretend we'd produce output, but cancellation is set so the
        # outer loop never hands us control.
        yield playground.GenerationToken(text="should-not-appear")
        yield playground.GenerationToken(done=True)

    monkeypatch.setattr(playground, "generate_stream", _slow_generate)
    cancel = threading.Event()
    cancel.set()
    turns = [
        multi_chat.MultiChatTurn(
            run_dict=_stub_run("a"),
            messages=({"role": "user", "content": "hi"},),
        ),
    ]
    frames = list(
        multi_chat.stream_turn(
            turns, runs_root=Path("/tmp"), cancel_event=cancel,
        )
    )
    assert all(f.text == "" for f in frames)
    assert frames[-1].done is True


def test_stream_turn_with_zero_turns_terminates_cleanly(monkeypatch):
    frames = list(multi_chat.stream_turn([], runs_root=Path("/tmp")))
    assert len(frames) == 1
    assert frames[0].done is True


def test_stream_turn_passes_per_adapter_messages_to_playground(monkeypatch):
    """Multi-turn correctness: each adapter must see its own message
    list (since their prior responses diverge). The orchestrator
    builds GenerationConfig.messages from the turn's history; verify
    the messages thread through unchanged."""
    captured: list[tuple[str, tuple]] = []

    def capture_generate(run_dict, gcfg, runs_root, cancel_event=None):
        captured.append((run_dict["id"], gcfg.messages))
        yield playground.GenerationToken(text="ok")
        yield playground.GenerationToken(done=True)

    monkeypatch.setattr(playground, "generate_stream", capture_generate)
    turns = [
        multi_chat.MultiChatTurn(
            run_dict=_stub_run("a"),
            messages=(
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1-from-a"},
                {"role": "user", "content": "Q2"},
            ),
        ),
        multi_chat.MultiChatTurn(
            run_dict=_stub_run("b"),
            messages=(
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1-from-b"},
                {"role": "user", "content": "Q2"},
            ),
        ),
    ]
    list(multi_chat.stream_turn(turns, runs_root=Path("/tmp")))
    assert len(captured) == 2
    assert captured[0][1][1]["content"] == "A1-from-a"
    assert captured[1][1][1]["content"] == "A1-from-b"
