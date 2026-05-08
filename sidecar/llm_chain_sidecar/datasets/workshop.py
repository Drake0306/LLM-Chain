"""Dataset workshop: parse pasted/raw text into rows, map onto a chat
schema, run a fixed pipeline of cleaners, and emit a JSONL chat file
the trainer can consume directly.

The sidecar exposes one endpoint (POST /api/datasets/build) that
calls into here. Splitting the logic out lets us test the parse +
clean pipeline without spinning up FastAPI, and lets the frontend
client share the same field-name conventions without re-implementing
them in TypeScript.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

InputFormat = Literal["csv", "tsv", "jsonl"]
TargetFormat = Literal["chat", "completion"]


@dataclass
class CleaningOptions:
    """Toggles for the post-mapping cleanup pipeline.

    Each toggle is applied in a fixed order so two runs over the same
    input produce the same output. Order matters because dedup after
    role-balance can drop different rows than dedup before it.
    """

    drop_empty: bool = True
    dedupe: bool = True
    role_balance: bool = True
    max_chars: int | None = None


@dataclass
class BuildStats:
    input_rows: int = 0
    dropped_empty: int = 0
    dropped_duplicate: int = 0
    dropped_role_violation: int = 0
    dropped_length: int = 0
    output_rows: int = 0

    def to_dict(self) -> dict:
        return {
            "input_rows": self.input_rows,
            "dropped_empty": self.dropped_empty,
            "dropped_duplicate": self.dropped_duplicate,
            "dropped_role_violation": self.dropped_role_violation,
            "dropped_length": self.dropped_length,
            "output_rows": self.output_rows,
        }


@dataclass
class SchemaMapping:
    """How to interpret each input row.

    Exactly one of the three (user/assistant) | (prompt/completion) |
    (chat) shapes is honoured per call. For chat-shaped input the row
    is expected to already carry a 'messages' key matching the
    JSONL_CHAT format — we just pass it through after cleaning.
    """

    target: TargetFormat = "chat"
    user_field: str | None = None
    assistant_field: str | None = None
    prompt_field: str | None = None
    completion_field: str | None = None
    # When set, treat the input rows as already chat-shaped and just
    # validate + clean them. Used for jsonl input that already has
    # 'messages'.
    passthrough_chat: bool = False


# --- parsing -----------------------------------------------------------


def parse_text(text: str, fmt: InputFormat) -> list[dict]:
    """Parse pasted text into row dicts.

    Returns an empty list for empty/whitespace-only input rather than
    raising — the route layer treats zero rows as a 400, but the parser
    itself stays tolerant so a UI-side preview doesn't flash an error
    every keystroke.
    """
    if not text or not text.strip():
        return []
    if fmt == "jsonl":
        rows: list[dict] = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Row {i}: not valid JSON ({e.msg} at col {e.colno})."
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Row {i}: each line must be a JSON object, got "
                    f"{type(obj).__name__}."
                )
            rows.append(obj)
        return rows
    if fmt in ("csv", "tsv"):
        delim = "," if fmt == "csv" else "\t"
        reader = csv.DictReader(io.StringIO(text), delimiter=delim)
        if not reader.fieldnames:
            raise ValueError(
                "Header row missing — first line must list column names."
            )
        return [dict(row) for row in reader]
    raise ValueError(f"Unknown input format: {fmt}")


# --- schema detection -------------------------------------------------


_USER_HINTS = ("user", "question", "input", "prompt", "instruction", "human")
_ASSISTANT_HINTS = ("assistant", "answer", "output", "response", "completion", "ai", "bot")


def detect_schema(rows: Iterable[dict]) -> SchemaMapping:
    """Best-effort schema sniff for CSV/TSV-style flat rows.

    Returns a mapping the UI can pre-fill. Falls back to a chat passthrough
    when the rows already look like JSONL chat (have a 'messages' key).
    The user can always override the suggestion in the workshop UI.
    """
    sample = next(iter(rows), None)
    if not isinstance(sample, dict) or not sample:
        return SchemaMapping(target="chat")
    if "messages" in sample:
        return SchemaMapping(target="chat", passthrough_chat=True)
    keys_lower = {k.lower(): k for k in sample.keys()}
    user = next((keys_lower[h] for h in _USER_HINTS if h in keys_lower), None)
    assistant = next(
        (keys_lower[h] for h in _ASSISTANT_HINTS if h in keys_lower), None
    )
    if user and assistant:
        # Pick "completion" target when the column names imply a flat
        # prompt/completion pair rather than a chat turn — same data
        # shape, but signals to the UI which terminology to lead with.
        prompt_like = user in {"prompt", "instruction"}
        completion_like = assistant in {"completion", "response"}
        if prompt_like and completion_like:
            return SchemaMapping(
                target="completion",
                prompt_field=user,
                completion_field=assistant,
            )
        return SchemaMapping(
            target="chat", user_field=user, assistant_field=assistant
        )
    return SchemaMapping(target="chat")


# --- mapping to chat shape -------------------------------------------


def _row_to_messages(
    row: dict, user_field: str, assistant_field: str
) -> dict | None:
    user = row.get(user_field)
    assistant = row.get(assistant_field)
    if user is None or assistant is None:
        return None
    user_str = str(user).strip()
    assistant_str = str(assistant).strip()
    # Either side empty means the row has nothing to teach the model —
    # an assistant response without a user prompt (or vice versa) breaks
    # the chat template's user→assistant alternation. Drop here so the
    # workshop never produces JSONL the trainer would silently mis-pad.
    if not user_str or not assistant_str:
        return None
    return {
        "messages": [
            {"role": "user", "content": user_str},
            {"role": "assistant", "content": assistant_str},
        ]
    }


def apply_schema(rows: list[dict], schema: SchemaMapping) -> list[dict]:
    """Convert flat rows to chat-shaped rows in-place style.

    Returns a new list — never mutates the input. Rows that can't be
    mapped (missing fields) are dropped silently here; the cleaner
    counts them under ``dropped_empty`` so the user sees the loss in
    the build summary.
    """
    if schema.passthrough_chat:
        return [r for r in rows if isinstance(r, dict) and "messages" in r]
    if schema.target == "chat":
        if not schema.user_field or not schema.assistant_field:
            raise ValueError(
                "Chat target needs user_field and assistant_field set."
            )
        mapped = []
        for r in rows:
            m = _row_to_messages(r, schema.user_field, schema.assistant_field)
            if m is not None:
                mapped.append(m)
        return mapped
    if schema.target == "completion":
        if not schema.prompt_field or not schema.completion_field:
            raise ValueError(
                "Completion target needs prompt_field and completion_field set."
            )
        mapped = []
        for r in rows:
            m = _row_to_messages(
                r, schema.prompt_field, schema.completion_field
            )
            if m is not None:
                mapped.append(m)
        return mapped
    raise ValueError(f"Unknown target: {schema.target}")


# --- cleaners ---------------------------------------------------------


def _row_text(row: dict) -> str:
    """Concatenate all message contents — used for hashing + length."""
    msgs = row.get("messages") or []
    parts: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
    return "\n".join(parts)


def _is_well_formed_chat(row: dict) -> bool:
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False
    for m in msgs:
        if not isinstance(m, dict):
            return False
        if "role" not in m or "content" not in m:
            return False
        if not isinstance(m["content"], str):
            return False
    return True


def _has_user_then_assistant(row: dict) -> bool:
    """Cheapest check that the conversation starts with user → assistant.

    The trainer's chat template breaks on assistant-first rows, and
    rows where every message has the same role aren't useful training
    signal anyway. We only enforce the prefix shape, not strict
    alternation past the first pair, since multi-turn rows often
    interleave system / tool messages legitimately.
    """
    msgs = row.get("messages") or []
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    # Skip a leading 'system' message — it's metadata, not the turn.
    if roles and roles[0] == "system":
        roles = roles[1:]
    if len(roles) < 2:
        return False
    return roles[0] == "user" and roles[1] == "assistant"


def clean(
    rows: list[dict], opts: CleaningOptions
) -> tuple[list[dict], BuildStats]:
    """Run the cleaning pipeline. Order is fixed so results are stable.

    1. drop_empty — rows where every message body trims to empty
    2. role_balance — drop rows whose conversation prefix isn't user→assistant
    3. max_chars — drop rows whose total content exceeds the cap
    4. dedupe — content-hash within the surviving set

    Stats record per-stage drop counts so the user can see which
    cleaner did the work. ``input_rows`` is the count entering this
    function; the workshop route adds the upstream-mapping drops back
    via ``dropped_empty`` before returning to the client.
    """
    stats = BuildStats(input_rows=len(rows))
    survivors: list[dict] = []
    for row in rows:
        if not _is_well_formed_chat(row):
            stats.dropped_empty += 1
            continue
        if opts.drop_empty and not _row_text(row).strip():
            stats.dropped_empty += 1
            continue
        if opts.role_balance and not _has_user_then_assistant(row):
            stats.dropped_role_violation += 1
            continue
        if opts.max_chars is not None and len(_row_text(row)) > opts.max_chars:
            stats.dropped_length += 1
            continue
        survivors.append(row)

    if opts.dedupe:
        seen: set[str] = set()
        deduped: list[dict] = []
        for row in survivors:
            key = _row_text(row)
            if key in seen:
                stats.dropped_duplicate += 1
                continue
            seen.add(key)
            deduped.append(row)
        survivors = deduped

    stats.output_rows = len(survivors)
    return survivors, stats


# --- output -----------------------------------------------------------


_NAME_RE = re.compile(r"[^a-z0-9_-]+")


def safe_filename(name: str) -> str:
    """Sanitise a user-supplied name into a stable slug.

    JSONL files end up listed in the dataset picker; characters that
    confuse filesystems (slashes, colons, spaces) get squashed to a
    single dash. An empty result falls back to ``dataset`` so we
    never write ``.jsonl`` with no stem.
    """
    s = (name or "").strip().lower()
    s = _NAME_RE.sub("-", s).strip("-")
    return s or "dataset"


def write_jsonl(rows: list[dict], path: Path) -> int:
    """Write rows as JSONL to ``path``. Returns bytes written.

    Atomic: stages all bytes into a sibling ``.tmp`` file, fsyncs,
    then ``os.replace`` to the final path. A crash mid-write leaves
    a stray ``.tmp`` rather than a half-written ``.jsonl`` that the
    loader would silently truncate at the last good line.
    """
    import os as _os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    written = 0
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                line = json.dumps(row, ensure_ascii=False) + "\n"
                f.write(line)
                written += len(line.encode("utf-8"))
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp, path)
    except Exception:
        # Best-effort cleanup so a failed write doesn't litter sibling
        # ``.tmp`` files for the user to manually delete.
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return written


def default_datasets_dir() -> Path:
    """Resolve ``~/.llm-chain/datasets`` honouring the test override env.

    Mirrors the runs-root override pattern in api.routes — tests set
    LLM_CHAIN_DATASETS_DIR to a tmp_path so they don't litter the
    user's home directory.
    """
    import os

    env = os.environ.get("LLM_CHAIN_DATASETS_DIR")
    if env:
        return Path(env)
    return Path.home() / ".llm-chain" / "datasets"
