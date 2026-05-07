import sys
from unittest.mock import MagicMock, patch

import pytest

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers.base import EventType


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_mlx_yields_start_step_done(tmp_path):
    from llm_chain_sidecar.trainers.mlx import MlxTrainer

    cfg = RunConfig(model_id="mlx-community/Qwen3-0.6B-4bit",
                    backend="mlx", technique="qlora",
                    dataset_path="ignored", epochs=1)
    trainer = MlxTrainer(cfg, output_dir=str(tmp_path))

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
