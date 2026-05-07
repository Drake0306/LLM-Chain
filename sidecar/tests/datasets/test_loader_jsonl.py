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
