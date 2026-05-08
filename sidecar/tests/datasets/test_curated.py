import json
from pathlib import Path

import pytest

from llm_chain_sidecar.datasets.curated import (
    CuratedEntry,
    _flatten_oasst_tree,
    _transform_conversations,
    _transform_instruction_output,
    _transform_messages,
    download_curated,
    find_entry,
    load_manifest,
    transform_rows,
)


# --- load_manifest ---------------------------------------------------


def test_load_manifest_ships_with_known_schemas():
    """Sanity check the in-package YAML — every entry's schema must
    be one we know how to transform, and every entry must carry an
    explicit license field."""
    entries = load_manifest()
    assert entries  # at least one entry shipped
    for e in entries:
        assert e.schema in {
            "instruction_output",
            "conversations",
            "oasst_tree",
            "messages",
        }
        assert e.license, f"{e.id}: missing license"
        assert e.hf_id, f"{e.id}: missing hf_id"


def test_load_manifest_rejects_unknown_schema(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "datasets:\n"
        "  - id: x\n"
        "    name: X\n"
        "    hf_id: foo/bar\n"
        "    license: MIT\n"
        "    schema: definitely_not_real\n"
    )
    with pytest.raises(ValueError, match="unknown schema"):
        load_manifest(p)


def test_load_manifest_rejects_duplicate_ids(tmp_path: Path):
    p = tmp_path / "dup.yaml"
    p.write_text(
        "datasets:\n"
        "  - id: a\n"
        "    name: A\n"
        "    hf_id: foo/bar\n"
        "    license: MIT\n"
        "    schema: messages\n"
        "  - id: a\n"
        "    name: A2\n"
        "    hf_id: foo/baz\n"
        "    license: MIT\n"
        "    schema: messages\n"
    )
    with pytest.raises(ValueError, match="duplicate id"):
        load_manifest(p)


def test_load_manifest_surfaces_missing_required_field(tmp_path: Path):
    p = tmp_path / "missing.yaml"
    p.write_text(
        "datasets:\n"
        "  - name: no-id-here\n"
        "    hf_id: foo/bar\n"
        "    license: MIT\n"
        "    schema: messages\n"
    )
    with pytest.raises(ValueError, match="missing required field"):
        load_manifest(p)


def test_find_entry():
    entries = load_manifest()
    sample = entries[0]
    assert find_entry(entries, sample.id) is sample
    assert find_entry(entries, "ghost-not-real") is None


# --- per-schema transforms -------------------------------------------


def test_instruction_output_concatenates_input_under_instruction():
    row = {
        "instruction": "Translate to French",
        "input": "Good morning",
        "response": "Bonjour",
    }
    out = _transform_instruction_output(row)
    assert out is not None
    assert out["messages"][0]["role"] == "user"
    assert "Translate to French" in out["messages"][0]["content"]
    assert "Good morning" in out["messages"][0]["content"]
    assert out["messages"][1]["content"] == "Bonjour"


def test_instruction_output_handles_dolly_response_field():
    """Dolly uses ``response``; Alpaca uses ``output``. Both should map
    to the assistant message."""
    out = _transform_instruction_output(
        {"instruction": "Q", "response": "A"}
    )
    assert out is not None
    assert out["messages"][1]["content"] == "A"


def test_instruction_output_drops_rows_with_empty_fields():
    assert _transform_instruction_output({"instruction": "", "response": "x"}) is None
    assert _transform_instruction_output({"instruction": "x", "response": ""}) is None


def test_conversations_maps_sharegpt_roles():
    row = {
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello"},
        ]
    }
    out = _transform_conversations(row)
    assert out is not None
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]


def test_conversations_drops_single_role_rows():
    row = {"conversations": [{"from": "human", "value": "Hi only"}]}
    assert _transform_conversations(row) is None


def test_conversations_keeps_system_messages():
    row = {
        "conversations": [
            {"from": "system", "value": "be helpful"},
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello"},
        ]
    }
    out = _transform_conversations(row)
    assert out is not None
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant"]


def test_messages_passthrough_validates_shape():
    row = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    out = _transform_messages(row)
    assert out is not None
    assert len(out["messages"]) == 2


def test_messages_passthrough_rejects_non_string_content():
    row = {
        "messages": [
            {"role": "user", "content": ["list", "not", "string"]},
            {"role": "assistant", "content": "ok"},
        ]
    }
    assert _transform_messages(row) is None


def test_oasst_tree_picks_highest_ranked_path():
    """Two replies to the same prompter; the lower-rank one should
    win (rank 0 is best in the OASST schema)."""
    rows = [
        {
            "message_id": "root",
            "parent_id": None,
            "role": "prompter",
            "lang": "en",
            "text": "Hello?",
        },
        {
            "message_id": "good",
            "parent_id": "root",
            "role": "assistant",
            "lang": "en",
            "text": "Best reply",
            "rank": 0,
        },
        {
            "message_id": "bad",
            "parent_id": "root",
            "role": "assistant",
            "lang": "en",
            "text": "Worse reply",
            "rank": 5,
        },
    ]
    out = _flatten_oasst_tree(rows)
    assert len(out) == 1
    assert out[0]["messages"][1]["content"] == "Best reply"


def test_oasst_tree_drops_non_english():
    rows = [
        {
            "message_id": "root",
            "parent_id": None,
            "role": "prompter",
            "lang": "de",
            "text": "Hallo?",
        },
        {
            "message_id": "rep",
            "parent_id": "root",
            "role": "assistant",
            "lang": "de",
            "text": "Hallo",
        },
    ]
    assert _flatten_oasst_tree(rows) == []


# --- download_curated -------------------------------------------------


def test_download_curated_writes_jsonl_and_returns_counts(tmp_path: Path, monkeypatch):
    """End-to-end: with the HF loader stubbed, the curated downloader
    should fetch via load_dataset, run the schema transform, write
    JSONL, and report (loaded, kept) counts."""
    entry = CuratedEntry(
        id="fake",
        name="Fake",
        hf_id="fake/dataset",
        description="",
        license="MIT",
        license_url="",
        size_rows=3,
        size_mb=0,
        schema="instruction_output",
        suitable_for=[],
    )

    fake_rows = [
        {"instruction": "Q1", "response": "A1"},
        {"instruction": "Q2", "response": "A2"},
        {"instruction": "", "response": "A3"},  # dropped
    ]

    def fake_load_dataset(hf_id, split="train"):
        return fake_rows

    import datasets as _hf_datasets

    monkeypatch.setattr(_hf_datasets, "load_dataset", fake_load_dataset)

    result = download_curated(entry, datasets_dir=tmp_path)
    assert Path(result.path).exists()
    assert result.rows_loaded == 3
    assert result.rows_kept == 2

    # JSONL must round-trip through the trainer's loader.
    from llm_chain_sidecar.datasets.loader import load_dataset as ours
    from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource

    loaded = ours(DatasetSource(format=DatasetFormat.JSONL_CHAT, path=result.path))
    assert len(loaded) == 2


def test_download_curated_refuses_to_overwrite(tmp_path: Path):
    entry = CuratedEntry(
        id="exists",
        name="X",
        hf_id="foo/bar",
        description="",
        license="MIT",
        license_url="",
        size_rows=0,
        size_mb=0,
        schema="messages",
    )
    (tmp_path / "exists.jsonl").write_text("placeholder\n")
    with pytest.raises(FileExistsError):
        download_curated(entry, datasets_dir=tmp_path)


def test_download_curated_raises_when_transform_keeps_zero_rows(
    tmp_path: Path, monkeypatch,
):
    """An empty post-transform output points at a real bug (HF schema
    drift, broken transform) and should surface clearly rather than
    write a zero-row JSONL the trainer would reject."""
    entry = CuratedEntry(
        id="empty",
        name="Empty",
        hf_id="foo/bar",
        description="",
        license="MIT",
        license_url="",
        size_rows=0,
        size_mb=0,
        schema="instruction_output",
    )

    def fake_load_dataset(hf_id, split="train"):
        return [{"instruction": "", "response": ""}]

    import datasets as _hf_datasets

    monkeypatch.setattr(_hf_datasets, "load_dataset", fake_load_dataset)

    with pytest.raises(ValueError, match="0 well-shaped rows"):
        download_curated(entry, datasets_dir=tmp_path)
