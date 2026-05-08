"""LoRA/QLoRA fine-tuning via ``mlx_lm lora`` on Apple Silicon.

The subprocess lifecycle, stdout pump, error reporting, and
GeneratorExit-safe cleanup all live in MlxSubprocessTrainer (see
trainers/_mlx_base.py). This module configures the two pieces that are
specific to the text trainer: which mlx tool to invoke, and how to map
each loader format onto the JSONL shapes mlx_lm understands.
"""
from llm_chain_sidecar.datasets import DatasetFormat

from ._mlx_base import MlxSubprocessTrainer


class MlxTrainer(MlxSubprocessTrainer):
    _module_name = "mlx_lm"
    _staged_dir_name = "_mlx_data"

    @property
    def _done_message(self) -> str:
        return f"Adapter saved to {self.output_dir}"

    def _validate_format(self, ds_format: DatasetFormat) -> None:
        if ds_format == DatasetFormat.JSONL_CHAT_VISION:
            # Vision rows belong on the mlx_vlm trainer; the API layer
            # should have routed here for text-only formats. Defense in
            # depth so a bad backend resolution doesn't silently train on
            # corrupted text.
            raise ValueError(
                "Vision-language datasets require the mlx_vlm backend, not mlx. "
                "Pick a vision model on the Models page."
            )

    def _row_to_mlx(self, row: dict, ds_format: DatasetFormat) -> dict:
        """Coerce a loader row into one of mlx_lm's two supported shapes.

        Chat datasets pass through (mlx_lm reads ``messages`` and applies
        the tokenizer chat template). Everything else gets emitted as
        ``{"text": ...}`` so mlx_lm's completion path picks it up.
        """
        if ds_format == DatasetFormat.JSONL_CHAT:
            return {"messages": row["messages"]}
        if ds_format == DatasetFormat.CSV:
            col = self.config.text_column or "text"
            if col not in row:
                raise ValueError(
                    f"CSV column '{col}' not found in row keys {sorted(row)}"
                )
            return {"text": str(row[col])}
        if ds_format == DatasetFormat.TEXT_DIR:
            return {"text": str(row.get("text", ""))}
        if ds_format == DatasetFormat.HF_HUB:
            # HF datasets vary by source — try a small priority list of
            # common text-bearing columns before falling back to the
            # user's text_column. Surface a clear error if nothing matches
            # so the user picks a column instead of getting a silent
            # empty corpus.
            preferred = self.config.text_column
            for col in (preferred, "text", "content", "input"):
                if col and col in row:
                    return {"text": str(row[col])}
            raise ValueError(
                f"Couldn't find a text column in HF Hub row {sorted(row)}; "
                "set 'text_column' on the dataset to point at the right field."
            )
        raise NotImplementedError(f"MLX staging doesn't handle {ds_format}")
