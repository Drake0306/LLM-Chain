import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers.base import EventType


def _write_tiny_jsonl(path: Path, n: int = 3) -> Path:
    """Write a chat-vision JSONL alongside placeholder image files. The
    loader resolves image paths relative to the JSONL's directory and
    validates each one exists, so the dummy files have to live next to the
    JSONL even though the staging code never opens them.
    """
    lines = []
    for i in range(n):
        img = path.parent / f"img{i}.png"
        img.write_bytes(b"")  # existence check only — staging never opens it
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


_VLM_KW = {"dataset_format": "jsonl_chat_vision"}


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_vlm_yields_start_step_done(tmp_path):
    from llm_chain_sidecar.trainers.mlx_vlm import MlxVlmTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(
        model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        backend="mlx_vlm", technique="lora",
        dataset_path=str(data), epochs=1, **_VLM_KW,
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
    with patch("llm_chain_sidecar.trainers._mlx_base.subprocess.Popen", return_value=fake_proc):
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
                    dataset_path=str(data), epochs=1, **_VLM_KW)
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
                    dataset_path=str(data), epochs=1, **_VLM_KW)
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
                    dataset_path=str(data), epochs=1, **_VLM_KW)
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))

    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = iter([b""])
    fake_proc.wait.return_value = 2
    with patch("llm_chain_sidecar.trainers._mlx_base.subprocess.Popen", return_value=fake_proc):
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
                    dataset_path=str(data), epochs=1, **_VLM_KW)
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))

    fake_lines = iter([
        b"ImportError: No module named 'mlx_vlm.lora'\n",
        b"",
    ])
    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = fake_lines
    fake_proc.wait.return_value = 1
    with patch("llm_chain_sidecar.trainers._mlx_base.subprocess.Popen", return_value=fake_proc):
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
                    dataset_path=str(data), epochs=1, **_VLM_KW)
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))

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
def test_mlx_vlm_stage_data_rejects_non_vision_format(tmp_path):
    """The text trainer should be picked for non-vision datasets. If a bad
    backend resolution lands a non-vision row here, fail clearly rather
    than try to copy bytes into a JSONL that mlx_vlm will misinterpret."""
    from llm_chain_sidecar.trainers.mlx_vlm import MlxVlmTrainer

    data = tmp_path / "chat.jsonl"
    data.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(
        model_id="m", backend="mlx_vlm", technique="lora",
        dataset_path=str(data), dataset_format="jsonl_chat", epochs=1,
    )
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))
    with pytest.raises(ValueError, match="jsonl_chat_vision"):
        trainer._stage_data()


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_vlm_stage_data_writes_test_jsonl_and_absolute_image_paths(tmp_path):
    """The previous staging copied the source JSONL bytes verbatim, so
    image paths stayed relative to the source file's directory and
    mlx_vlm — running from the staged dir — failed to find them. Now we
    route through the loader, which absolutizes every image path."""
    from llm_chain_sidecar.trainers.mlx_vlm import MlxVlmTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl", n=2)
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx_vlm", technique="lora",
                    dataset_path=str(data), epochs=1, **_VLM_KW)
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))
    staged = Path(trainer._stage_data())
    assert (staged / "test.jsonl").exists()

    rows = [
        json.loads(ln)
        for ln in (staged / "train.jsonl").read_text().splitlines()
        if ln.strip()
    ] + [
        json.loads(ln)
        for ln in (staged / "valid.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    image_parts = [
        part for r in rows for m in r["messages"]
        if isinstance(m["content"], list)
        for part in m["content"]
        if part.get("type") == "image"
    ]
    assert image_parts, "expected at least one image content part after staging"
    for part in image_parts:
        assert Path(part["path"]).is_absolute()
        assert Path(part["path"]).exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_vlm_uses_mlx_vlm_subcommand_form(tmp_path):
    from llm_chain_sidecar.trainers.mlx_vlm import MlxVlmTrainer

    data = _write_tiny_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx_vlm", technique="lora",
                    dataset_path=str(data), epochs=1, **_VLM_KW)
    trainer = MlxVlmTrainer(cfg, output_dir=str(out))

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

    assert "mlx_vlm.lora" not in captured_cmd
    assert "mlx_vlm" in captured_cmd
    idx = captured_cmd.index("mlx_vlm")
    assert captured_cmd[idx + 1] == "lora"


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
