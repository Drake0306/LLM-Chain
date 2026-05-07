import pytest

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers.base import EventType, Trainer, TrainingEvent


def test_trainer_is_abstract():
    with pytest.raises(TypeError):
        Trainer(  # type: ignore[abstract]
            RunConfig(model_id="m", backend="cpu", technique="lora",
                      dataset_path="/tmp/x", epochs=1),
            output_dir="/tmp/out",
        )


def test_event_construction():
    e = TrainingEvent(type=EventType.STEP, step=1, total_steps=10,
                      loss=2.3, lr=2e-4)
    assert e.step == 1
    assert e.loss == 2.3
