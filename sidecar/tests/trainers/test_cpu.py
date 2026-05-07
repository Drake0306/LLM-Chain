from unittest.mock import patch

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers import CpuTrainer, make_trainer
from llm_chain_sidecar.trainers.base import EventType


def test_make_trainer_cpu_returns_cpu_trainer(tmp_path):
    cfg = RunConfig(
        model_id="m", backend="cpu", technique="lora",
        dataset_path="ignored", epochs=1, batch_size=1,
    )
    trainer = make_trainer("cpu", cfg, str(tmp_path))
    assert isinstance(trainer, CpuTrainer)


def test_cpu_trainer_yields_start_then_steps_then_done(tmp_path):
    cfg = RunConfig(
        model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        backend="cpu", technique="lora",
        dataset_path="ignored", epochs=1, batch_size=1,
    )
    trainer = CpuTrainer(cfg, output_dir=str(tmp_path))

    fake_events = [
        {"step": 1, "loss": 2.5, "lr": 2e-4, "total_steps": 2},
        {"step": 2, "loss": 2.1, "lr": 2e-4, "total_steps": 2},
    ]
    with patch.object(trainer, "_run_training_loop", return_value=iter(fake_events)):
        events = list(trainer.train())

    assert events[0].type == EventType.START
    assert events[1].type == EventType.STEP and events[1].loss == 2.5
    assert events[2].type == EventType.STEP and events[2].loss == 2.1
    assert events[-1].type == EventType.DONE


def test_cpu_trainer_translates_download_events(tmp_path):
    cfg = RunConfig(
        model_id="m", backend="cpu", technique="lora",
        dataset_path="ignored", epochs=1, batch_size=1,
    )
    trainer = CpuTrainer(cfg, output_dir=str(tmp_path))

    raw = [
        {"type": "download", "bytes_done": 50, "bytes_total": 500, "desc": "tokenizer.json"},
        {"step": 1, "loss": 2.0, "lr": 2e-4, "total_steps": 1},
    ]
    with patch.object(trainer, "_run_training_loop", return_value=iter(raw)):
        events = list(trainer.train())

    download_events = [e for e in events if e.type == EventType.DOWNLOAD]
    assert len(download_events) == 1
    assert download_events[0].bytes_done == 50
    assert download_events[0].message == "tokenizer.json"


def test_cpu_trainer_emits_canceled_when_event_set(tmp_path):
    import threading
    cfg = RunConfig(
        model_id="m", backend="cpu", technique="lora",
        dataset_path="ignored", epochs=1, batch_size=1,
    )
    cancel = threading.Event()
    cancel.set()
    trainer = CpuTrainer(cfg, output_dir=str(tmp_path), cancel_event=cancel)

    with patch.object(trainer, "_run_training_loop", return_value=iter([])):
        events = list(trainer.train())

    assert events[-1].type == EventType.CANCELED
