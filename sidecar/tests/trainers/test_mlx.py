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
        b"Iter 10: Train loss 2.34, Learning Rate 1.0e-4\n",
        b"Iter 20: Train loss 1.98, Learning Rate 1.0e-4\n",
        b"",  # EOF
    ])
    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = fake_lines
    fake_proc.wait.return_value = 0
    with patch("llm_chain_sidecar.trainers.mlx.subprocess.Popen", return_value=fake_proc):
        events = list(trainer.train())

    assert events[0].type == EventType.START
    losses = [e.loss for e in events if e.type == EventType.STEP]
    assert losses == [2.34, 1.98]
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

    with patch("llm_chain_sidecar.trainers.mlx.subprocess.Popen", side_effect=fake_popen):
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
    with patch("llm_chain_sidecar.trainers.mlx.subprocess.Popen", return_value=fake_proc):
        events = list(trainer.train())

    err = events[-1]
    assert err.type == EventType.ERROR
    assert "exited 1" in err.message
    assert "Requested to train 16 layers" in err.message
    assert "Traceback" in err.message


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_stage_data_handles_single_row(tmp_path):
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl", n=1)
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="qlora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(out))
    staged = Path(trainer._stage_data())
    assert (staged / "train.jsonl").read_text().strip()
    assert (staged / "valid.jsonl").read_text().strip()
