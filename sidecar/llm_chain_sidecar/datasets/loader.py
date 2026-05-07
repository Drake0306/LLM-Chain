import json
from pathlib import Path

from .types import DatasetFormat, DatasetSource


def load_dataset(src: DatasetSource) -> list[dict]:
    if src.format == DatasetFormat.JSONL_CHAT:
        return _load_jsonl_chat(Path(src.path))
    raise NotImplementedError(f"Format {src.format} not yet implemented")


def _load_jsonl_chat(path: Path) -> list[dict]:
    rows: list[dict] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "messages" not in obj:
            raise ValueError(f"Row {i}: missing 'messages' key")
        if not isinstance(obj["messages"], list) or not obj["messages"]:
            raise ValueError(f"Row {i}: 'messages' must be a non-empty list")
        for j, m in enumerate(obj["messages"]):
            if "role" not in m or "content" not in m:
                raise ValueError(f"Row {i} msg {j}: missing role/content")
        rows.append(obj)
    return rows
