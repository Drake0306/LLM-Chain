import json
from pathlib import Path

import pytest

from llm_chain_sidecar.datasets.loader import load_dataset
from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource


def test_jsonl_chat_loads(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"},
                                 {"role": "assistant", "content": "hello"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "bye"},
                                   {"role": "assistant", "content": "goodbye"}]}) + "\n"
    )
    src = DatasetSource(format=DatasetFormat.JSONL_CHAT, path=str(p))
    ds = load_dataset(src)
    assert len(ds) == 2
    assert ds[0]["messages"][0]["role"] == "user"


def test_jsonl_chat_rejects_malformed(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"not_messages": []}\n')
    src = DatasetSource(format=DatasetFormat.JSONL_CHAT, path=str(p))
    with pytest.raises(ValueError, match="missing 'messages'"):
        load_dataset(src)


def test_jsonl_chat_invalid_json_points_at_row_and_file(tmp_path: Path):
    """A bad line should name the row number and file so the user can
    grep to the offending entry. Pre-fix, json.JSONDecodeError surfaced
    on its own with no file context — useless for a 10k-row dataset."""
    p = tmp_path / "data.jsonl"
    p.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n"
        + "this is not json\n"
    )
    with pytest.raises(ValueError, match="Row 2.*not valid JSON"):
        load_dataset(DatasetSource(format=DatasetFormat.JSONL_CHAT, path=str(p)))


def test_jsonl_chat_rejects_non_object_lines(tmp_path: Path):
    """A line that parses as a list / string / number is a JSON value but
    not an object with 'messages'. The previous code would IndexError or
    AttributeError on obj['messages'] without a hint the row was the wrong
    type."""
    p = tmp_path / "data.jsonl"
    p.write_text("[1, 2, 3]\n")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_dataset(DatasetSource(format=DatasetFormat.JSONL_CHAT, path=str(p)))


def test_jsonl_chat_surfaces_non_utf8_with_clear_error(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_bytes(b'{"messages": [{"role": "user", "content": "\xff\xfe"}]}\n')
    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_dataset(DatasetSource(format=DatasetFormat.JSONL_CHAT, path=str(p)))


def test_jsonl_chat_strips_utf8_bom_prefix(tmp_path: Path):
    """Windows tools (Notepad, Excel 'Save as CSV UTF-8') prepend the BOM
    \\xef\\xbb\\xbf. json.loads doesn't strip it and crashes on row 1
    with a misleading 'Expecting value'. utf-8-sig handles it
    transparently."""
    p = tmp_path / "bom.jsonl"
    body = (
        json.dumps({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]})
        + "\n"
    )
    p.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    rows = load_dataset(DatasetSource(format=DatasetFormat.JSONL_CHAT, path=str(p)))
    assert len(rows) == 1
    assert rows[0]["messages"][0]["content"] == "hi"
