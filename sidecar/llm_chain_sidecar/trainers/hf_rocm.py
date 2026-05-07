from .hf_cuda import HfCudaTrainer


_ROCM_NOT_VALIDATED_MESSAGE = (
    "ROCm trainer not yet validated on AMD hardware. "
    "The CUDA training path is structurally compatible with PyTorch+ROCm, "
    "but we have not yet exercised it on a real Radeon/Instinct GPU and so "
    "won't quietly run something that may produce broken adapters. "
    "If you have AMD hardware available, please report your experience at "
    "https://github.com/Drake0306/LLM-Chain/issues so we can validate and "
    "enable this backend."
)


class HfRocmTrainer(HfCudaTrainer):
    """Stub AMD/ROCm trainer.

    Subclasses HfCudaTrainer because PyTorch+ROCm reuses the CUDA API surface,
    so the actual training loop would only diverge in dtype quirks. Until
    someone runs this end-to-end on AMD silicon we refuse to instantiate
    rather than silently producing untrustworthy adapters.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        raise NotImplementedError(_ROCM_NOT_VALIDATED_MESSAGE)
