import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers.base import EventType


def _write_tiny_jsonl(path: Path, n: int = 3) -> Path:
    lines = [json.dumps({"messages": [
        {"role": "user", "content": f"q{i}"},
        {"role": "assistant", "content": f"a{i}"},
    ]}) for i in range(n)]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_yields_start_step_done(tmp_path):
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="mlx-community/Qwen3-0.6B-4bit",
                    backend="mlx", technique="qlora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(out))

    fake_lines = iter([
        b"Loading pretrained model\n",
        b"Iter 10: Train loss 2.34, Learning Rate 1.0e-4\n",
        b"Iter 20: Train loss 1.98, Learning Rate 1.0e-4\n",
        b"",  # EOF
    ])
    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = fake_lines
    fake_proc.wait.return_value = 0
    with patch("llm_chain_sidecar.trainers._mlx_base.subprocess.Popen", return_value=fake_proc):
        events = list(trainer.train())

    assert events[0].type == EventType.START
    losses = [e.loss for e in events if e.type == EventType.STEP]
    assert losses == [2.34, 1.98]
    # Non-loss progress lines are now forwarded as LOG events so the UI can
    # surface "Loading pretrained model" while the user waits for iter 1.
    log_messages = [e.message for e in events if e.type == EventType.LOG]
    assert "Loading pretrained model" in log_messages
    assert events[-1].type == EventType.DONE


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_stage_data_splits_into_train_and_valid(tmp_path):
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl", n=10)
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="qlora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(out))
    staged = Path(trainer._stage_data())
    train = (staged / "train.jsonl").read_text().splitlines()
    valid = (staged / "valid.jsonl").read_text().splitlines()
    assert len(train) == 9 and len(valid) == 1
    assert all(line.strip() for line in train + valid)


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_passes_num_layers_minus_one(tmp_path):
    """Default mlx_lm --num-layers is 16; tiny models (Pythia-70m has 6) raise
    ValueError before any step. Passing -1 trains all layers regardless of size."""
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="qlora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(out))

    captured_cmd = []
    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = iter([b""])
    fake_proc.wait.return_value = 0

    def fake_popen(cmd, **kw):
        captured_cmd.extend(cmd)
        return fake_proc

    with patch("llm_chain_sidecar.trainers._mlx_base.subprocess.Popen", side_effect=fake_popen):
        list(trainer.train())

    assert "--num-layers" in captured_cmd
    idx = captured_cmd.index("--num-layers")
    assert captured_cmd[idx + 1] == "-1"


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_includes_subprocess_output_in_error(tmp_path):
    """The trainer used to swallow every non-loss-line and surface only
    'mlx_lm exited N'. Users had no way to act. Now we keep the tail."""
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="qlora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(out))

    fake_lines = iter([
        b"Loading pretrained model\n",
        b"Traceback (most recent call last):\n",
        b"  File \"/x/lora.py\", line 226\n",
        b"ValueError: Requested to train 16 layers but the model only has 6 layers.\n",
        b"",
    ])
    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = fake_lines
    fake_proc.wait.return_value = 1
    with patch("llm_chain_sidecar.trainers._mlx_base.subprocess.Popen", return_value=fake_proc):
        events = list(trainer.train())

    err = events[-1]
    assert err.type == EventType.ERROR
    assert "exited 1" in err.message
    assert "Requested to train 16 layers" in err.message
    assert "Traceback" in err.message


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_stage_data_rejects_single_row_dataset(tmp_path):
    """With only one row the train/valid split has to overlap — eval loss
    becomes a copy of train loss, giving the user no signal. Better to
    fail loudly with an actionable ask than silently produce a bad
    fine-tune."""
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl", n=1)
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="qlora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(out))
    with pytest.raises(ValueError, match="at least 2"):
        trainer._stage_data()


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_stage_data_omits_empty_test_split(tmp_path):
    """mlx_lm's load_local_dataset reads test.jsonl unconditionally and
    feeds whatever it parses into create_dataset, which does ``sample =
    data[0]`` and crashes on []. The right contract is to NOT write
    test.jsonl when we have no test rows — load_subset's
    ``if not path.exists(): return []`` short-circuit handles the empty
    split cleanly. Writing an empty file defeats that short-circuit and
    surfaces in user logs as IndexError mid-load_dataset.
    """
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl", n=3)
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="qlora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(out))
    staged = Path(trainer._stage_data())
    assert (staged / "train.jsonl").exists()
    assert (staged / "valid.jsonl").exists()
    assert not (staged / "test.jsonl").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_stage_data_removes_stale_test_jsonl(tmp_path):
    """Defensive cleanup: a previous version of this code wrote an empty
    test.jsonl into the staged dir. If a re-run lands in the same dir
    (the user re-uses output_dir or rolls forward an old run), we have
    to actively remove that stale file so mlx_lm doesn't pick it up
    and crash. Simulate by pre-seeding the dir with an empty
    test.jsonl and verify staging clears it."""
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl", n=3)
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="qlora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(out))

    # Pre-seed the staged dir as if a buggy earlier run wrote test.jsonl.
    pre_staged = out / "_mlx_data"
    pre_staged.mkdir()
    (pre_staged / "test.jsonl").write_text("")

    staged = Path(trainer._stage_data())
    assert not (staged / "test.jsonl").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_stage_data_converts_csv_rows_to_text_jsonl(tmp_path):
    """The original staging copied bytes verbatim — a CSV file therefore
    landed in train.jsonl as raw CSV lines and mlx_lm crashed with
    JSONDecodeError on 'col1,col2,...'. Confirmed in the user's run logs.
    Now we route through the loader and emit `{"text": ...}` rows."""
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    csv = tmp_path / "data.csv"
    csv.write_text("body\nDemocracy is a form of government.\nFirst Moon landing in 1969.\n")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(
        model_id="m", backend="mlx", technique="lora",
        dataset_path=str(csv), dataset_format="csv", text_column="body", epochs=1,
    )
    trainer = MlxTrainer(cfg, output_dir=str(out))
    staged = Path(trainer._stage_data())
    train_lines = [
        json.loads(ln)
        for ln in (staged / "train.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    valid_lines = [
        json.loads(ln)
        for ln in (staged / "valid.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    all_rows = train_lines + valid_lines
    assert all_rows, "expected at least one staged row"
    assert all("text" in r for r in all_rows)
    assert "Democracy" in all_rows[0]["text"] or "Democracy" in all_rows[-1]["text"]


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_stage_data_converts_text_dir_to_text_jsonl(tmp_path):
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("alpha")
    (corpus / "b.txt").write_text("beta")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(
        model_id="m", backend="mlx", technique="lora",
        dataset_path=str(corpus), dataset_format="text_dir", epochs=1,
    )
    trainer = MlxTrainer(cfg, output_dir=str(out))
    staged = Path(trainer._stage_data())
    rows = [
        json.loads(ln)
        for ln in (staged / "train.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    rows += [
        json.loads(ln)
        for ln in (staged / "valid.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    texts = sorted(r["text"] for r in rows)
    assert texts == ["alpha", "beta"]


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_stage_data_passes_chat_messages_through(tmp_path):
    """JSONL chat rows should land in train.jsonl as-is (mlx_lm's chat
    path reads `messages` and applies the tokenizer chat template)."""
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl", n=3)
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(
        model_id="m", backend="mlx", technique="lora",
        dataset_path=str(data), dataset_format="jsonl_chat", epochs=1,
    )
    trainer = MlxTrainer(cfg, output_dir=str(out))
    staged = Path(trainer._stage_data())
    rows = [
        json.loads(ln)
        for ln in (staged / "train.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    assert rows
    assert all("messages" in r for r in rows)
    assert all("role" in m and "content" in m for r in rows for m in r["messages"])


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_stage_data_rejects_vision_format(tmp_path):
    """Vision rows belong on the mlx_vlm trainer. Defense in depth — if
    the route somehow lets a vision dataset through to the text trainer,
    fail loudly instead of silently mangling content lists."""
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
        + "\n"
    )
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(
        model_id="m", backend="mlx", technique="lora",
        dataset_path=str(data), dataset_format="jsonl_chat_vision", epochs=1,
    )
    trainer = MlxTrainer(cfg, output_dir=str(out))
    with pytest.raises(ValueError, match="mlx_vlm"):
        trainer._stage_data()


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_terminates_subprocess_on_generator_close(tmp_path):
    """Pre-fix, closing the trainer generator mid-run (SSE client
    disconnect) propagated GeneratorExit, which the inner Exception
    handler didn't catch, so proc.terminate() never ran and the mlx_lm
    OS process kept consuming RAM/CPU until the user manually killed it.
    Now there's a finally block that always reaps the subprocess.
    """
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="lora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(out))

    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    # Many lines so the consumer can abandon while still reading.
    fake_proc.stdout.readline.side_effect = iter(
        [b"Loading\n"] * 5 + [b"Iter 1: Train loss 1.0, Learning Rate 1e-4\n"] + [b""]
    )
    # Simulate "still running" the entire time so finally can observe and
    # call terminate() — first poll() returns None (alive), subsequent
    # poll()s after terminate() return 0.
    fake_proc.poll.side_effect = [None, 0]
    fake_proc.wait.return_value = 0

    with patch("llm_chain_sidecar.trainers._mlx_base.subprocess.Popen", return_value=fake_proc):
        gen = trainer.train()
        next(gen)  # START
        next(gen)  # whichever event comes first
        gen.close()  # SSE consumer disconnect

    fake_proc.terminate.assert_called_once()


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_kills_subprocess_when_terminate_times_out(tmp_path):
    """If terminate() doesn't reap the subprocess within the grace window,
    we escalate to kill() so the OS reclaims it. Pre-fix there was no
    timeout escalation and a stuck subprocess hung the trainer forever."""
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="lora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(out))

    import subprocess as sp_mod

    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    # Many lines so the consumer can interrupt with the trainer mid-loop.
    fake_proc.stdout.readline.side_effect = iter([b"Loading\n"] * 5 + [b""])
    fake_proc.poll.side_effect = [None]  # alive when finally checks
    # First wait() in the finally times out; second wait() after kill returns.
    fake_proc.wait.side_effect = [
        sp_mod.TimeoutExpired(cmd="mlx_lm", timeout=5),
        0,
    ]

    with patch("llm_chain_sidecar.trainers._mlx_base.subprocess.Popen", return_value=fake_proc):
        gen = trainer.train()
        next(gen)  # START — proc not yet spawned
        next(gen)  # advance into the read loop so the try/finally is active
        gen.close()  # SSE consumer disconnect

    fake_proc.terminate.assert_called_once()
    fake_proc.kill.assert_called_once()


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_uses_mlx_lm_subcommand_form(tmp_path):
    """Newer mlx_lm versions deprecate `python -m mlx_lm.lora` and emit a
    banner that ended up at the head of every failed-run error message
    in the user's logs. We invoke the subcommand form instead."""
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="lora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(out))

    captured_cmd: list[str] = []
    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = iter([b""])
    fake_proc.wait.return_value = 0

    def fake_popen(cmd, **kw):
        captured_cmd.extend(cmd)
        return fake_proc

    with patch("llm_chain_sidecar.trainers._mlx_base.subprocess.Popen", side_effect=fake_popen):
        list(trainer.train())

    # The deprecated form has 'mlx_lm.lora' as a single -m argument; the
    # supported form has 'mlx_lm' followed by the 'lora' subcommand.
    assert "mlx_lm.lora" not in captured_cmd
    assert "mlx_lm" in captured_cmd
    mlx_idx = captured_cmd.index("mlx_lm")
    assert captured_cmd[mlx_idx + 1] == "lora"
