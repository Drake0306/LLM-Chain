from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel


class License(str, Enum):
    APACHE_2_0 = "Apache-2.0"
    MIT = "MIT"
    LLAMA_COMMUNITY = "Llama-Community"
    GEMMA = "Gemma-ToU"
    FALCON = "Falcon-LLM-2.0"
    DEEPSEEK = "DeepSeek-Model-License"
    NVIDIA_OPEN = "NVIDIA-Open-Model"


class ModelEntry(BaseModel):
    id: str                        # HF Hub id, e.g. "Qwen/Qwen3-1.7B"
    name: str                      # Display name
    family: str                    # "Qwen3", "Mistral", ...
    params: int                    # parameter count
    license: License
    license_caveat: str | None = None
    modalities: list[str]          # ["text"], ["text", "image"]
    supports_lora: bool = True
    notes: str | None = None
    restricted: bool = False
    # Whether the tokenizer ships a chat_template. False for raw base
    # checkpoints (Pythia, Mistral-v0.3, base SmolLM2). The picker hides these
    # when the user has chosen a chat-format dataset, since mlx_lm / HF chat
    # training paths fail without a template.
    chat_capable: bool = True


class ModelRegistry:
    def __init__(self, entries: list[ModelEntry]):
        self._entries = list(entries)

    @classmethod
    def load_default(cls) -> "ModelRegistry":
        path = Path(__file__).parent / "data" / "allowlist.yaml"
        return cls.from_yaml(path)

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelRegistry":
        with path.open() as f:
            raw = yaml.safe_load(f)
        return cls(entries=[ModelEntry(**e) for e in raw["models"]])

    def entries(
        self,
        include_restricted: bool = False,
        required_modalities: list[str] | None = None,
        chat_capable_only: bool = False,
    ) -> list[ModelEntry]:
        out = (
            list(self._entries)
            if include_restricted
            else [e for e in self._entries if not e.restricted]
        )
        if required_modalities:
            needed = set(required_modalities)
            out = [e for e in out if needed.issubset(set(e.modalities))]
        if chat_capable_only:
            out = [e for e in out if e.chat_capable]
        return out

    def fitting_within(
        self,
        max_params: int,
        include_restricted: bool = False,
        required_modalities: list[str] | None = None,
        chat_capable_only: bool = False,
    ) -> list[ModelEntry]:
        return [
            e
            for e in self.entries(
                include_restricted=include_restricted,
                required_modalities=required_modalities,
                chat_capable_only=chat_capable_only,
            )
            if e.params <= max_params
        ]
