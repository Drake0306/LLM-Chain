"""F-C12: tests for the multi-adapter merge module.

Distinct from test_merge.py which covers gguf.merge_adapter (the
adapter-into-base fusion used by GGUF export). This file tests
exports/merge.py — the linear / TIES / DARE merge of N LoRA
adapters into a new adapter.
"""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from llm_chain_sidecar.exports.merge import (
    MergeInput,
    SUPPORTED_METHODS,
    merge_adapters,
)


def _write_adapter(
    dir_: Path,
    *,
    rank: int = 8,
    alpha: int = 16,
    base: str = "test/base",
    target_modules=None,
    fill: float = 1.0,
) -> Path:
    """Stage a fake PEFT adapter dir with a safetensors weights file
    and an adapter_config.json that matches what real PEFT saves.
    Tests build several of these to drive the merger."""
    dir_.mkdir(parents=True, exist_ok=True)
    cfg = {
        "base_model_name_or_path": base,
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": target_modules or ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
    }
    (dir_ / "adapter_config.json").write_text(json.dumps(cfg))
    save_file(
        {
            "base_model.layers.0.q_proj.lora_A.weight": torch.full(
                (rank, 32), fill,
            ),
            "base_model.layers.0.q_proj.lora_B.weight": torch.full(
                (32, rank), fill,
            ),
        },
        str(dir_ / "adapter_model.safetensors"),
    )
    return dir_


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_merge_writes_safetensors_and_config(tmp_path: Path, method):
    a = _write_adapter(tmp_path / "a", fill=1.0)
    b = _write_adapter(tmp_path / "b", fill=3.0)
    out = tmp_path / "merged"
    result = merge_adapters(
        [
            MergeInput(run_id="a", adapter_dir=a, weight=1.0),
            MergeInput(run_id="b", adapter_dir=b, weight=1.0),
        ],
        method=method,
        output_dir=out,
    )
    assert (out / "adapter_model.safetensors").exists()
    assert (out / "adapter_config.json").exists()
    assert (out / "merge.json").exists()
    assert result.method == method
    assert result.sources == ["a", "b"]
    assert result.tensor_count == 2


def test_linear_merge_produces_weighted_average(tmp_path: Path):
    """Equal weights: 1.0 + 3.0 → mean of 2.0. Locks the math so
    a future refactor breaking it surfaces immediately."""
    a = _write_adapter(tmp_path / "a", fill=1.0)
    b = _write_adapter(tmp_path / "b", fill=3.0)
    merge_adapters(
        [
            MergeInput(run_id="a", adapter_dir=a, weight=1.0),
            MergeInput(run_id="b", adapter_dir=b, weight=1.0),
        ],
        method="linear",
        output_dir=tmp_path / "out",
    )
    merged = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
    for t in merged.values():
        assert torch.allclose(t, torch.full_like(t, 2.0))


def test_linear_merge_respects_weight_ratios(tmp_path: Path):
    """Weights normalise; 1:3 → (1×1.0 + 3×5.0) / 4 = 4.0."""
    a = _write_adapter(tmp_path / "a", fill=1.0)
    b = _write_adapter(tmp_path / "b", fill=5.0)
    merge_adapters(
        [
            MergeInput(run_id="a", adapter_dir=a, weight=1.0),
            MergeInput(run_id="b", adapter_dir=b, weight=3.0),
        ],
        method="linear",
        output_dir=tmp_path / "out",
    )
    merged = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
    for t in merged.values():
        assert torch.allclose(t, torch.full_like(t, 4.0))


def test_merge_rejects_mismatched_rank(tmp_path: Path):
    a = _write_adapter(tmp_path / "a", rank=8)
    b = _write_adapter(tmp_path / "b", rank=16)
    with pytest.raises(ValueError, match="r"):
        merge_adapters(
            [
                MergeInput(run_id="a", adapter_dir=a, weight=1.0),
                MergeInput(run_id="b", adapter_dir=b, weight=1.0),
            ],
            method="linear",
            output_dir=tmp_path / "out",
        )


def test_merge_rejects_mismatched_base_model(tmp_path: Path):
    a = _write_adapter(tmp_path / "a", base="acme/model-a")
    b = _write_adapter(tmp_path / "b", base="acme/model-b")
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        merge_adapters(
            [
                MergeInput(run_id="a", adapter_dir=a, weight=1.0),
                MergeInput(run_id="b", adapter_dir=b, weight=1.0),
            ],
            method="linear",
            output_dir=tmp_path / "out",
        )


def test_merge_rejects_zero_weight_sum(tmp_path: Path):
    a = _write_adapter(tmp_path / "a")
    b = _write_adapter(tmp_path / "b")
    with pytest.raises(ValueError, match="must be > 0"):
        merge_adapters(
            [
                MergeInput(run_id="a", adapter_dir=a, weight=0.0),
                MergeInput(run_id="b", adapter_dir=b, weight=0.0),
            ],
            method="linear",
            output_dir=tmp_path / "out",
        )


def test_merge_rejects_single_input(tmp_path: Path):
    a = _write_adapter(tmp_path / "a")
    with pytest.raises(ValueError, match="at least 2"):
        merge_adapters(
            [MergeInput(run_id="a", adapter_dir=a, weight=1.0)],
            method="linear",
            output_dir=tmp_path / "out",
        )


def test_merge_rejects_unknown_method(tmp_path: Path):
    a = _write_adapter(tmp_path / "a")
    b = _write_adapter(tmp_path / "b")
    with pytest.raises(ValueError, match="Unknown merge method"):
        merge_adapters(
            [
                MergeInput(run_id="a", adapter_dir=a, weight=1.0),
                MergeInput(run_id="b", adapter_dir=b, weight=1.0),
            ],
            method="kitchen-sink",  # type: ignore[arg-type]
            output_dir=tmp_path / "out",
        )


def test_merge_rejects_missing_safetensors(tmp_path: Path):
    a = _write_adapter(tmp_path / "a")
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    (b_dir / "adapter_config.json").write_text(
        json.dumps({
            "base_model_name_or_path": "test/base",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
        })
    )
    with pytest.raises(FileNotFoundError, match="safetensors"):
        merge_adapters(
            [
                MergeInput(run_id="a", adapter_dir=a, weight=1.0),
                MergeInput(run_id="b", adapter_dir=b_dir, weight=1.0),
            ],
            method="linear",
            output_dir=tmp_path / "out",
        )


def test_target_modules_compared_set_wise(tmp_path: Path):
    """Order shouldn't matter; the validator normalises to a set."""
    a = _write_adapter(tmp_path / "a", target_modules=["q_proj", "v_proj"])
    b = _write_adapter(tmp_path / "b", target_modules=["v_proj", "q_proj"])
    merge_adapters(
        [
            MergeInput(run_id="a", adapter_dir=a, weight=1.0),
            MergeInput(run_id="b", adapter_dir=b, weight=1.0),
        ],
        method="linear",
        output_dir=tmp_path / "out",
    )


def test_dare_merge_is_deterministic_with_same_seed(tmp_path: Path):
    """Same recipe + same seed should produce a bit-identical merged
    adapter. Without seeding the RNG, two runs with the same recipe
    drift; the audit file would record an irreproducible artifact."""
    a = _write_adapter(tmp_path / "a", fill=1.0)
    b = _write_adapter(tmp_path / "b", fill=3.0)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    for out in (out1, out2):
        merge_adapters(
            [
                MergeInput(run_id="a", adapter_dir=a, weight=1.0),
                MergeInput(run_id="b", adapter_dir=b, weight=1.0),
            ],
            method="dare",
            output_dir=out,
            method_options={"drop_p": 0.5, "seed": 42},
        )
    t1 = load_file(str(out1 / "adapter_model.safetensors"))
    t2 = load_file(str(out2 / "adapter_model.safetensors"))
    for k in t1:
        assert torch.equal(t1[k], t2[k])


def test_dare_merge_differs_with_different_seeds(tmp_path: Path):
    """Sanity check the seed actually moves the output — otherwise
    the determinism test above could pass via accidental statelessness."""
    a = _write_adapter(tmp_path / "a", fill=1.0)
    b = _write_adapter(tmp_path / "b", fill=3.0)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    for out, seed in ((out1, 1), (out2, 2)):
        merge_adapters(
            [
                MergeInput(run_id="a", adapter_dir=a, weight=1.0),
                MergeInput(run_id="b", adapter_dir=b, weight=1.0),
            ],
            method="dare",
            output_dir=out,
            method_options={"drop_p": 0.5, "seed": seed},
        )
    t1 = load_file(str(out1 / "adapter_model.safetensors"))
    t2 = load_file(str(out2 / "adapter_model.safetensors"))
    different = any(not torch.equal(t1[k], t2[k]) for k in t1)
    assert different


def test_merge_target_modules_string_vs_list_rejected(tmp_path: Path):
    """An adapter saved with target_modules='all-linear' versus one
    saved with the expanded list ['q_proj', ...] — same logical
    target, different on-disk shape. The merger can't expand the
    string without loading the base model, so we surface the
    mismatch as a ValueError with a hint instead of silently
    accepting (which would later crash inside _merge_linear)."""
    a = _write_adapter(
        tmp_path / "a", target_modules=["q_proj", "v_proj"],
    )
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    (b_dir / "adapter_config.json").write_text(
        json.dumps({
            "base_model_name_or_path": "test/base",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": "all-linear",  # string form
        })
    )
    save_file(
        {
            "base_model.layers.0.q_proj.lora_A.weight": torch.zeros((8, 32)),
            "base_model.layers.0.q_proj.lora_B.weight": torch.zeros((32, 8)),
        },
        str(b_dir / "adapter_model.safetensors"),
    )
    with pytest.raises(ValueError, match="target_modules"):
        merge_adapters(
            [
                MergeInput(run_id="a", adapter_dir=a, weight=1.0),
                MergeInput(run_id="b", adapter_dir=b_dir, weight=1.0),
            ],
            method="linear",
            output_dir=tmp_path / "out",
        )


def test_merge_audit_file_records_recipe(tmp_path: Path):
    a = _write_adapter(tmp_path / "a")
    b = _write_adapter(tmp_path / "b")
    merge_adapters(
        [
            MergeInput(run_id="a", adapter_dir=a, weight=2.0),
            MergeInput(run_id="b", adapter_dir=b, weight=1.0),
        ],
        method="dare",
        output_dir=tmp_path / "out",
        method_options={"drop_p": 0.3},
    )
    audit = json.loads((tmp_path / "out" / "merge.json").read_text())
    assert audit["method"] == "dare"
    # Resolved options include the seed default (0) so the recipe is
    # reproducible from the audit alone.
    assert audit["method_options"]["drop_p"] == 0.3
    assert audit["method_options"]["seed"] == 0
    assert audit["sources"] == ["a", "b"]
    # Weights are normalised to sum to 1 — 2:1 → [0.667, 0.333].
    # raw_weights preserves what the user typed.
    assert audit["raw_weights"] == [2.0, 1.0]
    assert audit["weights"][0] > audit["weights"][1]
    assert abs(sum(audit["weights"]) - 1.0) < 1e-6
