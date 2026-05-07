import csv
import json
from pathlib import Path

from .types import DatasetFormat, DatasetSource


def load_dataset(src: DatasetSource) -> list[dict]:
    if src.format == DatasetFormat.JSONL_CHAT:
        return _load_jsonl_chat(Path(src.path))
    if src.format == DatasetFormat.JSONL_CHAT_VISION:
        return _load_jsonl_chat_vision(Path(src.path))
    if src.format == DatasetFormat.CSV:
        return _load_csv(Path(src.path), src.text_column)
    if src.format == DatasetFormat.TEXT_DIR:
        return _load_text_dir(Path(src.path))
    if src.format == DatasetFormat.HF_HUB:
        return _hf_load(src.hf_id, src.split)
    raise NotImplementedError(f"Format {src.format} not implemented")


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


def _load_jsonl_chat_vision(path: Path) -> list[dict]:
    """Load JSONL chat with image+text content arrays (OpenAI-style).

    Each row is ``{"messages": [{"role": ..., "content": [<parts>]}, ...]}``.
    Each content part is either ``{"type": "text", "text": "..."}`` or
    ``{"type": "image", "path": "..."}``. Image paths may be relative; we
    absolutize them against the JSONL's parent dir so the trainer can open
    them regardless of working directory. Plain string content (legacy chat
    rows) is normalized to ``[{"type": "text", "text": <str>}]`` so the
    trainer never has to branch on content shape.
    """
    rows: list[dict] = []
    base = path.parent
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
            content = m["content"]
            if isinstance(content, str):
                # Legacy chat row inside a vision dataset — normalize so all
                # downstream consumers see content arrays.
                m["content"] = [{"type": "text", "text": content}]
                continue
            if not isinstance(content, list) or not content:
                raise ValueError(
                    f"Row {i} msg {j}: 'content' must be a non-empty list or string"
                )
            for k, part in enumerate(content):
                ptype = part.get("type")
                if ptype == "text":
                    if "text" not in part:
                        raise ValueError(
                            f"Row {i} msg {j} part {k}: text part missing 'text'"
                        )
                elif ptype == "image":
                    if "path" not in part:
                        raise ValueError(
                            f"Row {i} msg {j} part {k}: image part missing 'path'"
                        )
                    img = Path(part["path"])
                    if not img.is_absolute():
                        img = (base / img).resolve()
                    if not img.exists():
                        raise FileNotFoundError(
                            f"Row {i} msg {j} part {k}: image not found at {img}"
                        )
                    part["path"] = str(img)
                else:
                    raise ValueError(
                        f"Row {i} msg {j} part {k}: unknown content type {ptype!r}"
                    )
        rows.append(obj)
    return rows


def _load_csv(path: Path, text_column: str | None) -> list[dict]:
    if not text_column:
        raise ValueError("CSV format requires text_column")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if text_column not in (reader.fieldnames or []):
            raise ValueError(f"column '{text_column}' not found in CSV")
        return list(reader)


def _load_text_dir(path: Path) -> list[dict]:
    return [{"text": p.read_text()} for p in sorted(path.glob("*.txt"))]


def _hf_load(hf_id: str, split: str) -> list[dict]:
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset(hf_id, split=split)
    return [dict(row) for row in ds]
