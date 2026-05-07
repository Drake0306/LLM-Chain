from unittest.mock import patch

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers.base import EventType
from llm_chain_sidecar.trainers.hf_cuda import HfCudaTrainer


def test_hf_cuda_yields_start_then_steps_then_done(tmp_path):
    cfg = RunConfig(model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
                    backend="cuda", technique="lora",
                    dataset_path="ignored", epochs=1, batch_size=1)
    trainer = HfCudaTrainer(cfg, output_dir=str(tmp_path))

    fake_callback_events = [
        {"step": 1, "loss": 2.0, "lr": 2e-4, "total_steps": 2},
        {"step": 2, "loss": 1.8, "lr": 2e-4, "total_steps": 2},
    ]
    with patch.object(trainer, "_run_training_loop", return_value=iter(fake_callback_events)):
        events = list(trainer.train())

    assert events[0].type == EventType.START
    assert events[1].type == EventType.STEP and events[1].loss == 2.0
    assert events[2].type == EventType.STEP and events[2].loss == 1.8
    assert events[-1].type == EventType.DONE
