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


class ModelRegistry(BaseModel):
    entries: list[ModelEntry]

    @classmethod
    def load_default(cls) -> "ModelRegistry":
        path = Path(__file__).parent / "data" / "allowlist.yaml"
        return cls.from_yaml(path)

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelRegistry":
        with path.open() as f:
            raw = yaml.safe_load(f)
        return cls(entries=[ModelEntry(**e) for e in raw["models"]])

    def fitting_within(self, max_params: int) -> list[ModelEntry]:
        return [e for e in self.entries if e.params <= max_params]
