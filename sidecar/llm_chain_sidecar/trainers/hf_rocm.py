import logging
import os

from .hf_cuda import HfCudaTrainer

log = logging.getLogger(__name__)

# Env var the user sets to opt into the experimental ROCm path. When unset
# the trainer refuses to instantiate. When set we still refuse QLoRA because
# bitsandbytes is CUDA-only; LoRA goes through under a loud warning.
EXPERIMENTAL_ENV_VAR = "LLM_CHAIN_ROCM_EXPERIMENTAL"


_ROCM_NOT_VALIDATED_MESSAGE = (
    "ROCm trainer not yet validated on AMD hardware. "
    "The CUDA training path is structurally compatible with PyTorch+ROCm, "
    "but we have not yet exercised it on a real Radeon/Instinct GPU and so "
    "won't quietly run something that may produce broken adapters. "
    f"If you have AMD hardware available and want to validate it locally, set "
    f"the {EXPERIMENTAL_ENV_VAR}=1 environment variable before launching the "
    "sidecar (LoRA only — see docs/amd-rocm-wsl2-setup.md). Either way, "
    "please report your experience at "
    "https://github.com/Drake0306/LLM-Chain/issues so we can validate and "
    "promote this backend out of experimental."
)


_ROCM_QLORA_NOT_SUPPORTED_MESSAGE = (
    "QLoRA on ROCm is not supported in the experimental unlock path because "
    "bitsandbytes is CUDA-only. Re-submit the run with technique=\"lora\". "
    "If you've installed an AMD bitsandbytes fork and want to try QLoRA "
    "anyway, please open an issue at "
    "https://github.com/Drake0306/LLM-Chain/issues with the install steps "
    "you used."
)


_ROCM_EXPERIMENTAL_ARMED_WARNING = (
    "AMD ROCm experimental path armed via %s — running unvalidated training. "
    "If the resulting adapter looks broken, please file an issue with the "
    "run directory contents."
)


def is_experimental_armed() -> bool:
    """Return True iff the user has opted into the experimental ROCm path.

    Read at every call so tests (and a sidecar restart in production) pick
    up the env var without a process restart between modes.
    """
    return os.environ.get(EXPERIMENTAL_ENV_VAR, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


class HfRocmTrainer(HfCudaTrainer):
    """Experimental AMD/ROCm trainer.

    Subclasses HfCudaTrainer because PyTorch+ROCm reuses the CUDA API surface,
    so the actual training loop only differs around the bitsandbytes (QLoRA)
    code path. By default we refuse to instantiate; setting
    ``LLM_CHAIN_ROCM_EXPERIMENTAL=1`` arms the LoRA path under a loud warning
    so users with AMD hardware can validate locally and report back.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        if not is_experimental_armed():
            raise NotImplementedError(_ROCM_NOT_VALIDATED_MESSAGE)

        super().__init__(*args, **kwargs)

        if getattr(self.config, "technique", None) == "qlora":
            raise NotImplementedError(_ROCM_QLORA_NOT_SUPPORTED_MESSAGE)

        log.warning(_ROCM_EXPERIMENTAL_ARMED_WARNING, EXPERIMENTAL_ENV_VAR)
