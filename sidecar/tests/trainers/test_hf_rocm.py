import pytest

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers import make_trainer
from llm_chain_sidecar.trainers.hf_cuda import HfCudaTrainer
from llm_chain_sidecar.trainers.hf_rocm import (
    EXPERIMENTAL_ENV_VAR,
    HfRocmTrainer,
    is_experimental_armed,
)


def _cfg(technique: str = "lora") -> RunConfig:
    return RunConfig(
        model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        backend="rocm",
        technique=technique,
        dataset_path="ignored",
        epochs=1,
        batch_size=1,
    )


def test_hf_rocm_trainer_is_a_cuda_subclass():
    # Subclassing the CUDA trainer is intentional — when ROCm is validated
    # the override surface should be small. Lock that in.
    assert issubclass(HfRocmTrainer, HfCudaTrainer)


def test_instantiating_hf_rocm_trainer_raises_when_flag_unset(monkeypatch, tmp_path):
    monkeypatch.delenv(EXPERIMENTAL_ENV_VAR, raising=False)
    with pytest.raises(NotImplementedError) as exc_info:
        HfRocmTrainer(_cfg(), output_dir=str(tmp_path))
    msg = str(exc_info.value)
    assert "not yet validated" in msg.lower()
    assert EXPERIMENTAL_ENV_VAR in msg
    assert "github.com/Drake0306/LLM-Chain/issues" in msg


def test_make_trainer_rocm_routes_to_stub_and_raises_when_flag_unset(monkeypatch, tmp_path):
    monkeypatch.delenv(EXPERIMENTAL_ENV_VAR, raising=False)
    with pytest.raises(NotImplementedError):
        make_trainer("rocm", _cfg(), output_dir=str(tmp_path))


@pytest.mark.parametrize("flag_value", ["1", "true", "TRUE", "yes", "on"])
def test_is_experimental_armed_truthy_values(monkeypatch, flag_value):
    monkeypatch.setenv(EXPERIMENTAL_ENV_VAR, flag_value)
    assert is_experimental_armed() is True


@pytest.mark.parametrize("flag_value", ["", "0", "false", "no", "off", "anythingelse"])
def test_is_experimental_armed_falsy_values(monkeypatch, flag_value):
    monkeypatch.setenv(EXPERIMENTAL_ENV_VAR, flag_value)
    # Anything that's not in the explicit truthy set is treated as off — we
    # don't want a typo'd value silently arming the trainer.
    if flag_value in ("1", "true", "yes", "on"):
        return
    assert is_experimental_armed() is False


def test_armed_lora_instantiates_without_raising(monkeypatch, tmp_path):
    monkeypatch.setenv(EXPERIMENTAL_ENV_VAR, "1")
    trainer = HfRocmTrainer(_cfg(technique="lora"), output_dir=str(tmp_path))
    # Inherits HfCudaTrainer construction — config + cancel_event should be set.
    assert trainer.config.backend == "rocm"
    assert trainer.config.technique == "lora"
    assert trainer.cancel_event is not None


def test_armed_qlora_still_refuses_with_bitsandbytes_message(monkeypatch, tmp_path):
    monkeypatch.setenv(EXPERIMENTAL_ENV_VAR, "1")
    with pytest.raises(NotImplementedError) as exc_info:
        HfRocmTrainer(_cfg(technique="qlora"), output_dir=str(tmp_path))
    msg = str(exc_info.value).lower()
    assert "qlora" in msg
    assert "bitsandbytes" in msg


def test_armed_path_logs_loud_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv(EXPERIMENTAL_ENV_VAR, "1")
    import logging
    with caplog.at_level(logging.WARNING, logger="llm_chain_sidecar.trainers.hf_rocm"):
        HfRocmTrainer(_cfg(technique="lora"), output_dir=str(tmp_path))
    # Surface the env var name in the log so users can grep their sidecar
    # log and see exactly why a run was allowed through.
    assert any(EXPERIMENTAL_ENV_VAR in r.getMessage() for r in caplog.records)
