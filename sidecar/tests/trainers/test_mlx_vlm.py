import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers.base import EventType


def _write_tiny_jsonl(path: Path, n: int = 3) -> Path:
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "image", "path": f"img{i}.png"},
                            {"type": "text", "text": f"q{i}"},
                        ]},
                        {"role": "assistant", "content": f"a{i}"},
                    ]
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_vlm_yields_start_step_done(tmp_path):
    from llm_chain_sidecar.trainers.mlx_vlm import MlxVlmTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(
        model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        backend="mlx_vlm", technique="lora",
        dataset_path=str(data), epochs=1,
    )
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))

    fake_lines = iter([
        b"Iter 10: Train loss 3.45, Learning Rate 1.0e-4\n",
        b"Iter 20: Train loss 2.98, Learning Rate 1.0e-4\n",
        b"",
    ])
    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = fake_lines
    fake_proc.wait.return_value = 0
    with patch("llm_chain_sidecar.trainers.mlx_vlm.subprocess.Popen", return_value=fake_proc):
        events = list(trainer.train())

    assert events[0].type == EventType.START
    losses = [e.loss for e in events if e.type == EventType.STEP]
    assert losses == [3.45, 2.98]
    assert events[-1].type == EventType.DONE


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_vlm_stage_data_splits_train_valid(tmp_path):
    from llm_chain_sidecar.trainers.mlx_vlm import MlxVlmTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl", n=10)
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx_vlm", technique="lora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))
    staged = Path(trainer._stage_data())
    train = (staged / "train.jsonl").read_text().splitlines()
    valid = (staged / "valid.jsonl").read_text().splitlines()
    assert len(train) == 9 and len(valid) == 1


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_vlm_stage_data_handles_single_row(tmp_path):
    from llm_chain_sidecar.trainers.mlx_vlm import MlxVlmTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl", n=1)
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx_vlm", technique="lora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))
    staged = Path(trainer._stage_data())
    assert (staged / "train.jsonl").read_text().strip()
    assert (staged / "valid.jsonl").read_text().strip()


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_vlm_emits_error_on_nonzero_exit(tmp_path):
    from llm_chain_sidecar.trainers.mlx_vlm import MlxVlmTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx_vlm", technique="lora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))

    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = iter([b""])
    fake_proc.wait.return_value = 2
    with patch("llm_chain_sidecar.trainers.mlx_vlm.subprocess.Popen", return_value=fake_proc):
        events = list(trainer.train())

    assert events[-1].type == EventType.ERROR
    assert "exited 2" in events[-1].message


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_vlm_includes_subprocess_output_in_error(tmp_path):
    from llm_chain_sidecar.trainers.mlx_vlm import MlxVlmTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx_vlm", technique="lora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))

    fake_lines = iter([
        b"ImportError: No module named 'mlx_vlm.lora'\n",
        b"",
    ])
    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = fake_lines
    fake_proc.wait.return_value = 1
    with patch("llm_chain_sidecar.trainers.mlx_vlm.subprocess.Popen", return_value=fake_proc):
        events = list(trainer.train())

    assert events[-1].type == EventType.ERROR
    assert "ImportError" in events[-1].message


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_vlm_passes_num_layers_minus_one(tmp_path):
    from llm_chain_sidecar.trainers.mlx_vlm import MlxVlmTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx_vlm", technique="lora",
                    dataset_path=str(data), epochs=1)
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))

    captured_cmd = []
    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = iter([b""])
    fake_proc.wait.return_value = 0

    def fake_popen(cmd, **kw):
        captured_cmd.extend(cmd)
        return fake_proc

    with patch("llm_chain_sidecar.trainers.mlx_vlm.subprocess.Popen", side_effect=fake_popen):
        list(trainer.train())

    assert "--num-layers" in captured_cmd
    idx = captured_cmd.index("--num-layers")
    assert captured_cmd[idx + 1] == "-1"


def test_make_trainer_mlx_vlm_returns_mlx_vlm_trainer():
    if sys.platform != "darwin":
        pytest.skip("MLX is macOS-only")
    from llm_chain_sidecar.trainers import MlxVlmTrainer, make_trainer

    cfg = RunConfig(
        model_id="m", backend="mlx_vlm", technique="lora",
        dataset_path="ignored", epochs=1,
    )
    trainer = make_trainer("mlx_vlm", cfg, "/tmp/x")
    assert isinstance(trainer, MlxVlmTrainer)
