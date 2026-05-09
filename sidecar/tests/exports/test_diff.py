import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from llm_chain_sidecar.exports.diff import diff_adapters


def _write_adapter(
    dir_: Path,
    *,
    rank: int = 4,
    base: str = "test/base",
    fill: float = 1.0,
    extra_keys: dict | None = None,
) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base,
                "r": rank,
                "lora_alpha": rank * 2,
                "target_modules": ["q_proj"],
            }
        )
    )
    tensors = {
        "base_model.layers.0.q_proj.lora_A.weight": torch.full(
            (rank, 16), fill,
        ),
        "base_model.layers.0.q_proj.lora_B.weight": torch.full(
            (16, rank), fill,
        ),
    }
    if extra_keys:
        tensors.update(extra_keys)
    save_file(tensors, str(dir_ / "adapter_model.safetensors"))
    return dir_


def test_diff_matched_layers_returns_frobenius(tmp_path: Path):
    """Both adapters all-ones vs all-twos: each delta tensor is ones,
    so ||delta||_F = sqrt(numel). Locks the math against a refactor."""
    a = _write_adapter(tmp_path / "a", fill=1.0)
    b = _write_adapter(tmp_path / "b", fill=2.0)
    result = diff_adapters(a, b)
    assert result.summary["matched_count"] == 2
    # Two tensors of shape (4, 16): numel = 64 each. ||1||_F = 8.
    for layer in result.layers:
        assert abs(layer.frobenius - 8.0) < 1e-5
        assert layer.abs_max == 1.0


def test_diff_sorts_by_frobenius_descending(tmp_path: Path):
    """The heatmap shows highest-change layers at the top, so the
    sort order matters. Build adapters where A's q_proj differs by
    2.0 and B's lora_B differs by 1.0; q_proj should land first."""
    a = _write_adapter(tmp_path / "a", fill=1.0)
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    (b_dir / "adapter_config.json").write_text(
        (a / "adapter_config.json").read_text()
    )
    save_file(
        {
            "base_model.layers.0.q_proj.lora_A.weight": torch.full(
                (4, 16), 3.0,  # delta of 2.0
            ),
            "base_model.layers.0.q_proj.lora_B.weight": torch.full(
                (16, 4), 2.0,  # delta of 1.0
            ),
        },
        str(b_dir / "adapter_model.safetensors"),
    )
    result = diff_adapters(a, b_dir)
    # First entry (largest Frobenius) should be the lora_A.weight.
    assert result.layers[0].key.endswith("lora_A.weight")


def test_diff_records_summary_stats(tmp_path: Path):
    a = _write_adapter(tmp_path / "a", fill=1.0)
    b = _write_adapter(tmp_path / "b", fill=2.0)
    result = diff_adapters(a, b)
    assert result.summary["max_frobenius"] > 0
    assert result.summary["mean_frobenius"] > 0
    assert result.summary["unmatched_keys"]["only_a"] == []
    assert result.summary["unmatched_keys"]["only_b"] == []


def test_diff_rejects_mismatched_base_model(tmp_path: Path):
    a = _write_adapter(tmp_path / "a", base="acme/model-a")
    b = _write_adapter(tmp_path / "b", base="acme/model-b")
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        diff_adapters(a, b)


def test_diff_rejects_mismatched_rank(tmp_path: Path):
    a = _write_adapter(tmp_path / "a", rank=4)
    b = _write_adapter(tmp_path / "b", rank=8)
    with pytest.raises(ValueError, match="'r'"):
        diff_adapters(a, b)


def test_diff_reports_unmatched_keys(tmp_path: Path):
    """If A has a key B doesn't (or vice versa), the diff records it
    under unmatched_keys so the user knows why some weights aren't
    represented in the heatmap."""
    a = _write_adapter(
        tmp_path / "a",
        extra_keys={
            "base_model.layers.1.v_proj.lora_A.weight": torch.zeros((4, 16)),
        },
    )
    b = _write_adapter(tmp_path / "b")
    result = diff_adapters(a, b)
    assert "base_model.layers.1.v_proj.lora_A.weight" in (
        result.summary["unmatched_keys"]["only_a"]
    )


def test_diff_zero_delta_when_identical(tmp_path: Path):
    a = _write_adapter(tmp_path / "a", fill=2.0)
    b = _write_adapter(tmp_path / "b", fill=2.0)
    result = diff_adapters(a, b)
    for layer in result.layers:
        assert layer.frobenius == 0.0
        assert layer.abs_max == 0.0


def test_diff_raises_on_missing_safetensors(tmp_path: Path):
    a = _write_adapter(tmp_path / "a")
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    (b_dir / "adapter_config.json").write_text(
        (a / "adapter_config.json").read_text()
    )
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        diff_adapters(a, b_dir)
