import json
from pathlib import Path

import pytest

from llm_chain_sidecar.datasets.loader import load_dataset
from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vision"


def test_loads_sample_with_image_and_text_parts():
    rows = load_dataset(
        DatasetSource(format=DatasetFormat.JSONL_CHAT_VISION, path=str(FIXTURE_DIR / "sample.jsonl"))
    )
    assert len(rows) == 2
    user = rows[0]["messages"][0]
    assert user["role"] == "user"
    parts = user["content"]
    assert parts[0]["type"] == "image"
    assert parts[0]["path"].endswith("a.png")
    assert Path(parts[0]["path"]).exists()
    assert parts[1] == {"type": "text", "text": "What color is this?"}


def test_relative_image_paths_resolve_against_jsonl_parent():
    rows = load_dataset(
        DatasetSource(format=DatasetFormat.JSONL_CHAT_VISION, path=str(FIXTURE_DIR / "sample.jsonl"))
    )
    img_path = Path(rows[0]["messages"][0]["content"][0]["path"])
    assert img_path.is_absolute()
    assert img_path.parent == FIXTURE_DIR


def test_string_content_is_normalized_to_text_part(tmp_path: Path):
    # Second row in the fixture has assistant content as a plain string.
    rows = load_dataset(
        DatasetSource(format=DatasetFormat.JSONL_CHAT_VISION, path=str(FIXTURE_DIR / "sample.jsonl"))
    )
    assistant = rows[1]["messages"][1]
    assert assistant["content"] == [{"type": "text", "text": "Blue."}]


def test_absolute_image_paths_pass_through(tmp_path: Path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)  # not a valid image but exists
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "path": str(img)},
                        {"type": "text", "text": "x"},
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
                ]
            }
        )
        + "\n"
    )
    rows = load_dataset(DatasetSource(format=DatasetFormat.JSONL_CHAT_VISION, path=str(jsonl)))
    assert rows[0]["messages"][0]["content"][0]["path"] == str(img)


def test_missing_image_raises_with_clear_message(tmp_path: Path):
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "path": "missing.png"},
                        {"type": "text", "text": "x"},
                    ]},
                    {"role": "assistant", "content": "y"},
                ]
            }
        )
        + "\n"
    )
    with pytest.raises(FileNotFoundError, match="image not found at"):
        load_dataset(DatasetSource(format=DatasetFormat.JSONL_CHAT_VISION, path=str(jsonl)))


def test_unknown_content_type_raises(tmp_path: Path):
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        json.dumps(
            {"messages": [
                {"role": "user", "content": [{"type": "audio", "path": "x.wav"}]},
                {"role": "assistant", "content": "y"},
            ]}
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="unknown content type 'audio'"):
        load_dataset(DatasetSource(format=DatasetFormat.JSONL_CHAT_VISION, path=str(jsonl)))


def test_text_part_missing_text_raises(tmp_path: Path):
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        json.dumps(
            {"messages": [
                {"role": "user", "content": [{"type": "text"}]},
                {"role": "assistant", "content": "y"},
            ]}
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="text part missing"):
        load_dataset(DatasetSource(format=DatasetFormat.JSONL_CHAT_VISION, path=str(jsonl)))
