import pytest

from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers import make_trainer
from llm_chain_sidecar.trainers.hf_cuda import HfCudaTrainer
from llm_chain_sidecar.trainers.hf_rocm import HfRocmTrainer


def _cfg() -> RunConfig:
    return RunConfig(
        model_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        backend="rocm",
        technique="lora",
        dataset_path="ignored",
        epochs=1,
        batch_size=1,
    )


def test_hf_rocm_trainer_is_a_cuda_subclass():
    # Subclassing the CUDA trainer is intentional — when ROCm is validated
    # the override surface should be small. Lock that in.
    assert issubclass(HfRocmTrainer, HfCudaTrainer)


def test_instantiating_hf_rocm_trainer_raises_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError) as exc_info:
        HfRocmTrainer(_cfg(), output_dir=str(tmp_path))
    msg = str(exc_info.value)
    assert "not yet validated" in msg.lower()
    assert "github.com/Drake0306/LLM-Chain/issues" in msg


def test_make_trainer_rocm_routes_to_stub_and_raises(tmp_path):
    with pytest.raises(NotImplementedError):
        make_trainer("rocm", _cfg(), output_dir=str(tmp_path))
