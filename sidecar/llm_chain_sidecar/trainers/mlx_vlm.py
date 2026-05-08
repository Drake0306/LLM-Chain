"""LoRA fine-tuning via ``mlx_vlm lora`` on Apple Silicon.

See trainers/_mlx_base.py for the shared subprocess infrastructure. This
module only configures the VLM-specific bits: target the ``mlx_vlm``
package, accept only the chat-vision format, and pass loader rows
through unchanged (the chat-vision loader has already absolutized image
paths and validated the message shape).
"""
from llm_chain_sidecar.datasets import DatasetFormat

from ._mlx_base import MlxSubprocessTrainer


class MlxVlmTrainer(MlxSubprocessTrainer):
    _module_name = "mlx_vlm"
    _staged_dir_name = "_mlx_vlm_data"

    @property
    def _done_message(self) -> str:
        return f"VLM adapter saved to {self.output_dir}"

    def _validate_format(self, ds_format: DatasetFormat) -> None:
        if ds_format != DatasetFormat.JSONL_CHAT_VISION:
            raise ValueError(
                f"mlx_vlm trainer only handles 'jsonl_chat_vision'; "
                f"got '{self.config.dataset_format}'."
            )

    def _row_to_mlx(self, row: dict, ds_format: DatasetFormat) -> dict:
        # The chat-vision loader has already done the heavy lifting:
        # validated message shape, normalised string content into a text
        # part, absolutized image paths against the source JSONL's parent.
        # Pass through unchanged.
        return {"messages": row["messages"]}
