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


def test_hf_cuda_translates_download_events(tmp_path):
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="ignored", epochs=1, batch_size=1)
    trainer = HfCudaTrainer(cfg, output_dir=str(tmp_path))

    raw = [
        {"type": "download", "bytes_done": 100, "bytes_total": 1000, "desc": "model.safetensors"},
        {"type": "download", "bytes_done": 1000, "bytes_total": 1000, "desc": "model.safetensors"},
        {"step": 1, "loss": 2.0, "lr": 2e-4, "total_steps": 1},
    ]
    with patch.object(trainer, "_run_training_loop", return_value=iter(raw)):
        events = list(trainer.train())

    download_events = [e for e in events if e.type == EventType.DOWNLOAD]
    assert len(download_events) == 2
    assert download_events[0].bytes_done == 100
    assert download_events[0].bytes_total == 1000
    assert download_events[0].message == "model.safetensors"
    assert download_events[1].bytes_done == 1000
