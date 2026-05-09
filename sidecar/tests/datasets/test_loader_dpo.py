import json
from pathlib import Path

import pytest

from llm_chain_sidecar.datasets.loader import load_dataset
from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource


def _src(p: Path) -> DatasetSource:
    return DatasetSource(format=DatasetFormat.JSONL_DPO, path=str(p))


def test_jsonl_dpo_loads_well_formed_rows(tmp_path: Path):
    p = tmp_path / "pref.jsonl"
    p.write_text(
        json.dumps({
            "prompt": "What's the capital of France?",
            "chosen": "Paris.",
            "rejected": "I don't know.",
        }) + "\n"
        + json.dumps({
            "prompt": "Translate 'hello' to Spanish.",
            "chosen": "Hola.",
            "rejected": "Hello in Spanish is hello.",
        }) + "\n"
    )
    rows = load_dataset(_src(p))
    assert len(rows) == 2
    assert rows[0]["prompt"] == "What's the capital of France?"
    assert rows[0]["chosen"] == "Paris."
    assert rows[1]["rejected"].startswith("Hello")


def test_jsonl_dpo_rejects_missing_field(tmp_path: Path):
    p = tmp_path / "broken.jsonl"
    p.write_text(json.dumps({"prompt": "x", "chosen": "y"}) + "\n")
    with pytest.raises(ValueError, match="missing required field 'rejected'"):
        load_dataset(_src(p))


def test_jsonl_dpo_rejects_empty_field(tmp_path: Path):
    """DPO loss is undefined when one side is empty — surface here so
    the user fixes the data instead of seeing a NaN loss curve."""
    p = tmp_path / "empty.jsonl"
    p.write_text(
        json.dumps({"prompt": "x", "chosen": "", "rejected": "y"}) + "\n"
    )
    with pytest.raises(ValueError, match="non-empty string"):
        load_dataset(_src(p))


def test_jsonl_dpo_rejects_non_string_field(tmp_path: Path):
    p = tmp_path / "wrong-type.jsonl"
    p.write_text(
        json.dumps({"prompt": "x", "chosen": ["list"], "rejected": "y"}) + "\n"
    )
    with pytest.raises(ValueError, match="non-empty string"):
        load_dataset(_src(p))


def test_jsonl_dpo_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "blanks.jsonl"
    p.write_text(
        json.dumps({"prompt": "a", "chosen": "b", "rejected": "c"}) + "\n"
        + "\n\n  \n"
        + json.dumps({"prompt": "d", "chosen": "e", "rejected": "f"}) + "\n"
    )
    assert len(load_dataset(_src(p))) == 2


def test_jsonl_dpo_invalid_json_points_at_row(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text(
        json.dumps({"prompt": "x", "chosen": "y", "rejected": "z"}) + "\n"
        + "not json at all\n"
    )
    with pytest.raises(ValueError, match="Row 2.*not valid JSON"):
        load_dataset(_src(p))
