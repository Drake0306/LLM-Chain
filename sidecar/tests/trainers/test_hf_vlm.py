from unittest.mock import patch

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers import HfVlmTrainer, make_trainer
from llm_chain_sidecar.trainers.base import EventType


def test_make_trainer_cuda_vlm_returns_hf_vlm_trainer(tmp_path):
    cfg = RunConfig(
        model_id="m", backend="cuda_vlm", technique="lora",
        dataset_path="ignored", epochs=1, batch_size=1,
    )
    trainer = make_trainer("cuda_vlm", cfg, str(tmp_path))
    assert isinstance(trainer, HfVlmTrainer)


def test_hf_vlm_yields_start_then_steps_then_done(tmp_path):
    cfg = RunConfig(
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        backend="cuda_vlm", technique="lora",
        dataset_path="ignored", epochs=1, batch_size=1,
    )
    trainer = HfVlmTrainer(cfg, output_dir=str(tmp_path))

    fake_events = [
        {"step": 1, "loss": 3.4, "lr": 2e-4, "total_steps": 2},
        {"step": 2, "loss": 2.9, "lr": 2e-4, "total_steps": 2},
    ]
    with patch.object(trainer, "_run_training_loop", return_value=iter(fake_events)):
        events = list(trainer.train())

    assert events[0].type == EventType.START
    assert events[1].type == EventType.STEP and events[1].loss == 3.4
    assert events[2].type == EventType.STEP and events[2].loss == 2.9
    assert events[-1].type == EventType.DONE


def test_hf_vlm_translates_download_events(tmp_path):
    cfg = RunConfig(
        model_id="m", backend="cuda_vlm", technique="lora",
        dataset_path="ignored", epochs=1, batch_size=1,
    )
    trainer = HfVlmTrainer(cfg, output_dir=str(tmp_path))

    raw = [
        {"type": "download", "bytes_done": 100, "bytes_total": 1000, "desc": "preprocessor_config.json"},
        {"type": "download", "bytes_done": 1000, "bytes_total": 1000, "desc": "preprocessor_config.json"},
        {"step": 1, "loss": 3.0, "lr": 2e-4, "total_steps": 1},
    ]
    with patch.object(trainer, "_run_training_loop", return_value=iter(raw)):
        events = list(trainer.train())

    download_events = [e for e in events if e.type == EventType.DOWNLOAD]
    assert len(download_events) == 2
    assert download_events[0].bytes_done == 100
    assert download_events[1].bytes_done == 1000
    assert download_events[0].message == "preprocessor_config.json"


def test_hf_vlm_emits_canceled_when_event_set(tmp_path):
    import threading
    cfg = RunConfig(
        model_id="m", backend="cuda_vlm", technique="lora",
        dataset_path="ignored", epochs=1, batch_size=1,
    )
    cancel = threading.Event()
    cancel.set()
    trainer = HfVlmTrainer(cfg, output_dir=str(tmp_path), cancel_event=cancel)

    with patch.object(trainer, "_run_training_loop", return_value=iter([])):
        events = list(trainer.train())

    assert events[-1].type == EventType.CANCELED


def test_hf_vlm_surfaces_loop_exceptions_as_error_events(tmp_path):
    cfg = RunConfig(
        model_id="m", backend="cuda_vlm", technique="lora",
        dataset_path="ignored", epochs=1, batch_size=1,
    )
    trainer = HfVlmTrainer(cfg, output_dir=str(tmp_path))

    def boom():
        raise RuntimeError("CUDA out of memory")
        yield  # unreachable, makes this a generator

    with patch.object(trainer, "_run_training_loop", side_effect=boom):
        events = list(trainer.train())

    assert events[-1].type == EventType.ERROR
    assert "CUDA out of memory" in events[-1].message
