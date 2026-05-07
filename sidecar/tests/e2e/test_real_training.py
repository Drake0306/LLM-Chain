import sys
from pathlib import Path

import pytest
import torch

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers import make_trainer
from llm_chain_sidecar.trainers.base import EventType

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.jsonl"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_lora_step_on_cuda(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("datasets")
    cfg = RunConfig(
        model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        backend="cuda", technique="lora",
        dataset_path=str(FIXTURE),
        epochs=1, batch_size=1, lora_rank=4, lora_alpha=8,
    )
    trainer = make_trainer("cuda", cfg, str(tmp_path))
    events = list(trainer.train())
    types = [e.type for e in events]
    assert EventType.START in types
    assert EventType.STEP in types
    assert EventType.DONE in types
    losses = [e.loss for e in events if e.type == EventType.STEP and e.loss is not None]
    assert len(losses) >= 1


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_real_lora_step_on_mlx(tmp_path):
    pytest.importorskip("mlx_lm")
    cfg = RunConfig(
        model_id="mlx-community/Qwen3-0.6B-4bit",
        backend="mlx", technique="qlora",
        dataset_path=str(FIXTURE),
        epochs=1, batch_size=1,
    )
    trainer = make_trainer("mlx", cfg, str(tmp_path))
    events = list(trainer.train())
    types = [e.type for e in events]
    assert EventType.START in types
    assert EventType.DONE in types
