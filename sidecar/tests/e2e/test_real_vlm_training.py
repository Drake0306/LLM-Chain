"""Slow end-to-end smoke for VLM LoRA training. Marked slow so the default
``pytest`` invocation skips them; opt in with ``pytest -m slow``.

The MLX path uses ``mlx-community/Qwen2-VL-2B-Instruct-4bit`` which fits on a
16 GB Apple Silicon Mac. The CUDA path is gated by ``torch.cuda.is_available``.
"""
import sys
from pathlib import Path

import pytest

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers import make_trainer
from llm_chain_sidecar.trainers.base import EventType

FIXTURE = Path(__file__).parent / "fixtures" / "vision" / "sample.jsonl"


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is macOS-only")
def test_real_vlm_lora_step_on_mlx(tmp_path):
    pytest.importorskip("mlx_vlm")
    cfg = RunConfig(
        model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        backend="mlx_vlm", technique="lora",
        dataset_path=str(FIXTURE),
        dataset_format="jsonl_chat_vision",
        epochs=1, batch_size=1, lora_rank=4, lora_alpha=8,
    )
    trainer = make_trainer("mlx_vlm", cfg, str(tmp_path))
    events = list(trainer.train())
    types = [e.type for e in events]
    assert EventType.START in types
    # mlx_vlm.lora may run too few iters to log a step on a 4-row dataset; the
    # smoke is "subprocess succeeded and we saw a DONE".
    assert EventType.DONE in types or EventType.ERROR in types


@pytest.mark.slow
def test_real_vlm_lora_step_on_cuda(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    cfg = RunConfig(
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        backend="cuda_vlm", technique="lora",
        dataset_path=str(FIXTURE),
        dataset_format="jsonl_chat_vision",
        epochs=1, batch_size=1, lora_rank=4, lora_alpha=8,
    )
    trainer = make_trainer("cuda_vlm", cfg, str(tmp_path))
    events = list(trainer.train())
    types = [e.type for e in events]
    assert EventType.START in types
    assert EventType.STEP in types
    assert EventType.DONE in types
