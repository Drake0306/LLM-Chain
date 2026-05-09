import csv
import json
from pathlib import Path

from .types import DatasetFormat, DatasetSource


def make_source(
    fmt: DatasetFormat | str,
    identifier: str,
    text_column: str | None = None,
    split: str = "train",
) -> DatasetSource:
    """Build a DatasetSource from the trainer's flat config shape.

    Trainers carry a single ``dataset_path`` string in RunConfig. For local
    formats it's a filesystem path; for ``HF_HUB`` it's the dataset id. This
    helper routes the identifier to the right field on DatasetSource so the
    loader sees the value where it expects it (HF Hub used to silently break
    because the loader read ``src.hf_id`` while the trainer set ``src.path``).
    """
    fmt = DatasetFormat(fmt) if not isinstance(fmt, DatasetFormat) else fmt
    if fmt == DatasetFormat.HF_HUB:
        return DatasetSource(format=fmt, hf_id=identifier, split=split, text_column=text_column)
    return DatasetSource(format=fmt, path=identifier, text_column=text_column)


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
    if src.format == DatasetFormat.JSONL_DPO:
        return _load_jsonl_dpo(Path(src.path))
    raise NotImplementedError(f"Format {src.format} not implemented")


def _read_text_safely(path: Path) -> str:
    """Read a text file, surfacing a clear error if it isn't UTF-8.

    Default ``Path.read_text`` raises ``UnicodeDecodeError`` with a hex
    offset that doesn't tell the user which file is at fault. We re-raise
    as ValueError with the file path so the route layer can pass it on.

    Reading with ``utf-8-sig`` instead of plain ``utf-8`` quietly strips
    the byte-order-mark prefix that Windows tools (Notepad, Excel "Save
    as CSV UTF-8") insert at the start of files. Without this strip,
    ``json.loads("\\ufeff{...}")`` crashes on row 1 with "Expecting
    value" — a message the user has no way to act on. Standard UTF-8
    files without a BOM are a strict subset of utf-8-sig, so this is
    safe for the common case.
    """
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"{path} is not valid UTF-8 (bad byte at offset {e.start}). "
            "Re-save the file as UTF-8 and try again."
        ) from e


def _load_jsonl_chat(path: Path) -> list[dict]:
    rows: list[dict] = []
    for i, line in enumerate(_read_text_safely(path).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Row {i} in {path.name} is not valid JSON: {e.msg} "
                f"(col {e.colno})"
            ) from e
        if not isinstance(obj, dict):
            raise ValueError(
                f"Row {i} in {path.name}: each line must be a JSON object, "
                f"got {type(obj).__name__}"
            )
        if "messages" not in obj:
            raise ValueError(f"Row {i}: missing 'messages' key")
        if not isinstance(obj["messages"], list) or not obj["messages"]:
            raise ValueError(f"Row {i}: 'messages' must be a non-empty list")
        for j, m in enumerate(obj["messages"]):
            if not isinstance(m, dict):
                raise ValueError(
                    f"Row {i} msg {j}: must be an object, got {type(m).__name__}"
                )
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
    for i, line in enumerate(_read_text_safely(path).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Row {i} in {path.name} is not valid JSON: {e.msg} "
                f"(col {e.colno})"
            ) from e
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
    """Load every ``.txt`` file under ``path`` (recursively).

    Originally this was a top-level ``glob("*.txt")``. Users with corpora
    organised into subfolders silently saw zero rows and got a confusing
    "No rows found" error from the staging step. Recursing matches the
    "drop a folder of files" mental model.
    """
    files = sorted(path.rglob("*.txt"))
    rows: list[dict] = []
    for p in files:
        if not p.is_file():
            continue
        rows.append({"text": _read_text_safely(p)})
    return rows


def _load_jsonl_dpo(path: Path) -> list[dict]:
    """Load DPO preference pairs (F-C10).

    Each row must carry ``prompt`` (the user-side input the model was
    asked), ``chosen`` (the preferred assistant response), and
    ``rejected`` (the dispreferred one). All three are required strings.

    The format is intentionally flat — TRL's DPOTrainer accepts the
    same shape directly with no further normalisation. Empty strings
    are rejected because DPO loss is undefined when one side is
    empty (the implicit "reward" gradient collapses).
    """
    rows: list[dict] = []
    for i, line in enumerate(_read_text_safely(path).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Row {i} in {path.name} is not valid JSON: {e.msg} "
                f"(col {e.colno})"
            ) from e
        if not isinstance(obj, dict):
            raise ValueError(
                f"Row {i} in {path.name}: each line must be a JSON object, "
                f"got {type(obj).__name__}"
            )
        for key in ("prompt", "chosen", "rejected"):
            if key not in obj:
                raise ValueError(
                    f"Row {i} in {path.name}: missing required field "
                    f"{key!r}. DPO format needs prompt/chosen/rejected."
                )
            if not isinstance(obj[key], str) or not obj[key].strip():
                raise ValueError(
                    f"Row {i} in {path.name}: field {key!r} must be a "
                    "non-empty string."
                )
        rows.append({
            "prompt": obj["prompt"],
            "chosen": obj["chosen"],
            "rejected": obj["rejected"],
        })
    return rows


def _hf_load(hf_id: str, split: str) -> list[dict]:
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset(hf_id, split=split)
    return [dict(row) for row in ds]
