"""Curated dataset manifest + downloader (F-B6).

Each entry in ``curated.yaml`` lists an HF Hub dataset the sidecar can
fetch on click — name, description, license, size, and a schema hint
that says how to map each row onto the JSONL chat format the trainer
consumes. The downloader pulls via ``datasets.load_dataset``, applies
the per-entry schema transform, and writes a normalised JSONL file
under the same datasets dir the workshop uses, so the dataset picker
surfaces everything in one place.

Why YAML manifest + per-entry transforms instead of "just upload the
HF id and use hf_hub format": the curated entries promise a known-good
JSONL chat shape regardless of what HF returns. Some datasets (Alpaca-
style instruction/output, OpenAssistant tree) need normalisation
before the trainer's loader can read them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

import yaml

from .workshop import default_datasets_dir, write_jsonl


@dataclass
class CuratedEntry:
    id: str
    name: str
    hf_id: str
    description: str
    license: str
    license_url: str
    size_rows: int
    size_mb: int
    schema: str
    suitable_for: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "hf_id": self.hf_id,
            "description": self.description,
            "license": self.license,
            "license_url": self.license_url,
            "size_rows": self.size_rows,
            "size_mb": self.size_mb,
            "schema": self.schema,
            "suitable_for": list(self.suitable_for),
        }


_KNOWN_SCHEMAS = {"instruction_output", "conversations", "oasst_tree", "messages"}

# Manifest entry ids become filenames under the datasets dir, so they
# need to be filesystem-safe. The pattern matches the schema we'd
# accept for any user-facing slug: lowercase letters / digits, plus
# dashes / dots / underscores in the body. Without this, an id like
# ``../../etc/something`` would land the JSONL outside the configured
# datasets root — a defence-in-depth concern even though the shipped
# manifest is trusted.
import re as _re

_ID_RE = _re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def load_manifest(path: Path | None = None) -> list[CuratedEntry]:
    """Parse ``curated.yaml`` into typed entries.

    The manifest ships inside the package, so the default path resolves
    via importlib.resources rather than relying on a CWD. Tests can
    pass an explicit path to a temp manifest to exercise edge cases
    without re-shipping the curated set.

    Schema strings are validated against ``_KNOWN_SCHEMAS`` — an entry
    with an unknown schema would silently produce malformed JSONL on
    download, which the trainer would reject downstream with a
    confusing error. Better to fail manifest-load.
    """
    if path is None:
        with resources.as_file(
            resources.files(__package__) / "curated.yaml",
        ) as p:
            text = Path(p).read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    out: list[CuratedEntry] = []
    for raw_entry in raw.get("datasets", []) or []:
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"curated.yaml: each entry must be a mapping, got {type(raw_entry).__name__}"
            )
        try:
            entry = CuratedEntry(
                id=raw_entry["id"],
                name=raw_entry["name"],
                hf_id=raw_entry["hf_id"],
                description=raw_entry.get("description", ""),
                license=raw_entry["license"],
                license_url=raw_entry.get("license_url", ""),
                size_rows=int(raw_entry.get("size_rows", 0)),
                size_mb=int(raw_entry.get("size_mb", 0)),
                schema=raw_entry["schema"],
                suitable_for=list(raw_entry.get("suitable_for", [])),
            )
        except KeyError as e:
            raise ValueError(
                f"curated.yaml entry missing required field: {e.args[0]!r}"
            ) from e
        if entry.schema not in _KNOWN_SCHEMAS:
            raise ValueError(
                f"curated.yaml: entry {entry.id!r} has unknown schema "
                f"{entry.schema!r}; expected one of {sorted(_KNOWN_SCHEMAS)}"
            )
        if not _ID_RE.match(entry.id):
            raise ValueError(
                f"curated.yaml: entry id {entry.id!r} must match "
                "[a-z0-9][a-z0-9._-]* — ids become filenames under the "
                "datasets dir and need to be filesystem-safe."
            )
        out.append(entry)
    # Detect duplicate ids early — two entries with the same slug
    # would race at download time over the same output file.
    seen: set[str] = set()
    for e in out:
        if e.id in seen:
            raise ValueError(f"curated.yaml: duplicate id {e.id!r}")
        seen.add(e.id)
    return out


def find_entry(entries: list[CuratedEntry], entry_id: str) -> CuratedEntry | None:
    return next((e for e in entries if e.id == entry_id), None)


# --- per-schema row transforms ---------------------------------------


def _normalise(value: Any) -> str:
    """Coerce a HF dataset value to a stripped string. HF can hand back
    None / numeric / list types depending on the dataset; the chat
    template expects strings."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_normalise(v) for v in value if v is not None)
    return str(value).strip()


def _transform_instruction_output(row: dict) -> dict | None:
    """Alpaca / Dolly: ``{instruction, [input,] response|output}`` →
    one (user, assistant) pair. ``input`` is concatenated under the
    instruction with a separator when present, matching the original
    Alpaca format the model authors trained against.
    """
    instruction = _normalise(row.get("instruction"))
    extra = _normalise(row.get("input") or row.get("context"))
    response = _normalise(row.get("response") or row.get("output"))
    if not instruction or not response:
        return None
    user = instruction if not extra else f"{instruction}\n\n{extra}"
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
        ]
    }


_HUMAN_FROMS = {"human", "user"}
_ASSISTANT_FROMS = {"gpt", "assistant", "bot", "ai"}


def _transform_conversations(row: dict) -> dict | None:
    """ShareGPT / SlimOrca: ``{conversations: [{from, value}, ...]}``
    → ``{messages: [{role, content}]}``. Drops the row if no
    user message survives or only one role is present."""
    convs = row.get("conversations")
    if not isinstance(convs, list):
        return None
    messages: list[dict] = []
    for c in convs:
        if not isinstance(c, dict):
            continue
        sender = (c.get("from") or "").lower().strip()
        content = _normalise(c.get("value"))
        if not content:
            continue
        if sender in _HUMAN_FROMS:
            role = "user"
        elif sender in _ASSISTANT_FROMS:
            role = "assistant"
        elif sender == "system":
            role = "system"
        else:
            continue
        messages.append({"role": role, "content": content})
    roles = {m["role"] for m in messages}
    if "user" not in roles or "assistant" not in roles:
        return None
    return {"messages": messages}


def _transform_messages(row: dict) -> dict | None:
    """Passthrough for already-chat-shaped rows (rare but cheap to
    support — saves writing a no-op transform per such dataset)."""
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    cleaned: list[dict] = []
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant", "system"):
            return None
        if not isinstance(content, str):
            return None
        cleaned.append({"role": role, "content": content})
    return {"messages": cleaned}


def _flatten_oasst_tree(rows: Iterable[dict]) -> list[dict]:
    """OpenAssistant ships a forest of message trees; each row is one
    node with ``parent_id`` pointing to its parent. We pick English
    rows, follow each leaf back to its root, and emit one chat
    conversation per root with the highest cumulative rank.

    The transformation keeps all ranks in mind: when a node has
    multiple replies, the one with the highest ``rank`` (or the
    earliest sibling if ranks are missing) wins. This matches the
    typical "best path" view of the dataset.
    """
    by_id: dict[str, dict] = {}
    children: dict[str | None, list[dict]] = {}
    for r in rows:
        if r.get("lang") and r["lang"] != "en":
            continue
        msg_id = r.get("message_id")
        if not msg_id:
            continue
        by_id[msg_id] = r
        children.setdefault(r.get("parent_id"), []).append(r)

    def _rank(node: dict) -> tuple[float, str]:
        # Lower rank wins (rank 0 is the "best"); fall back to a high
        # number when missing so unranked siblings sort last. Rank is
        # a float in real OASST data — we used to coerce to int which
        # collapsed near-ties unfairly. Tiebreaker is ``created_date``
        # (an ISO string on the upstream schema; lex-sorting works as
        # a chronology proxy because ISO-8601 sorts chronologically).
        rank = node.get("rank")
        rank_value = float(rank) if rank is not None else 1e9
        created = node.get("created_date") or ""
        return (rank_value, str(created))

    out: list[dict] = []
    # Roots are nodes whose parent_id is null/missing.
    roots = children.get(None, []) + children.get("", [])
    for root in roots:
        if root.get("role") != "prompter":
            continue
        cur: dict | None = root
        path: list[dict] = []
        while cur is not None:
            path.append(cur)
            replies = children.get(cur["message_id"], [])
            if not replies:
                break
            # Sort children by rank; lower wins.
            replies = sorted(replies, key=_rank)
            cur = replies[0]
        if len(path) < 2:
            continue
        msgs: list[dict] = []
        for node in path:
            role = "user" if node.get("role") == "prompter" else "assistant"
            content = _normalise(node.get("text"))
            if not content:
                continue
            msgs.append({"role": role, "content": content})
        if len(msgs) >= 2:
            out.append({"messages": msgs})
    return out


def transform_rows(entry: CuratedEntry, rows: Iterable[dict]) -> list[dict]:
    """Apply the entry's schema transform to a sequence of HF rows.

    Tree-structured schemas (oasst_tree) consume the whole iterable
    in one pass; per-row schemas iterate. Returns the list of
    well-shaped chat rows; malformed rows are dropped silently and
    the caller compares the count against the entry's advertised
    size_rows so the user sees the loss in the download summary.
    """
    if entry.schema == "oasst_tree":
        return _flatten_oasst_tree(rows)
    transformer: Any
    if entry.schema == "instruction_output":
        transformer = _transform_instruction_output
    elif entry.schema == "conversations":
        transformer = _transform_conversations
    elif entry.schema == "messages":
        transformer = _transform_messages
    else:
        raise ValueError(f"unknown schema {entry.schema!r}")
    out: list[dict] = []
    for row in rows:
        chat = transformer(row)
        if chat is not None:
            out.append(chat)
    return out


# --- download orchestrator -------------------------------------------


@dataclass
class DownloadResult:
    """What :func:`download_curated` returns to the route layer.

    ``path`` is where the JSONL landed; ``rows_loaded`` is the count
    HF returned; ``rows_kept`` is what survived the schema transform.
    The user sees the gap in the UI as "downloaded N, kept M
    after normalisation".
    """

    path: str
    rows_loaded: int
    rows_kept: int


def download_curated(
    entry: CuratedEntry,
    *,
    datasets_dir: Path | None = None,
    split: str = "train",
) -> DownloadResult:
    """Fetch ``entry.hf_id`` via ``datasets.load_dataset``, transform,
    and write the resulting JSONL chat file.

    The filename is the entry id + ``.jsonl`` so the dataset picker
    can refer back to it later without ambiguity. We refuse to
    overwrite an existing file — if the user wants to re-download
    they can delete the old file first; saves a "wait, did this
    download succeed or did it just no-op my old broken file?"
    debug session.
    """
    out_dir = (datasets_dir or default_datasets_dir()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f"{entry.id}.jsonl").resolve()
    # Defence-in-depth: even with the load_manifest id regex, refuse
    # to write outside the configured datasets dir. Catches any future
    # path of the entry id through that ends up with traversal.
    try:
        out_path.relative_to(out_dir)
    except ValueError as e:
        raise ValueError(
            f"refusing to write outside {out_dir}: resolved path "
            f"{out_path} escapes the configured datasets dir."
        ) from e
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists. Delete it on disk if you "
            "want to re-download."
        )

    # Lazy import — keeps this module cheap to load when no curated
    # download is in flight (and avoids a hard datasets dep at
    # import time on hosts that never use this surface).
    from datasets import load_dataset as hf_load_dataset

    if entry.schema == "oasst_tree":
        # OpenAssistant needs the full split to flatten trees — no
        # streaming. Rely on HF's local cache for warm reads.
        ds = hf_load_dataset(entry.hf_id, split=split)
        rows = [dict(r) for r in ds]
    else:
        ds = hf_load_dataset(entry.hf_id, split=split)
        rows = [dict(r) for r in ds]

    rows_loaded = len(rows)
    chat_rows = transform_rows(entry, rows)
    if not chat_rows:
        raise ValueError(
            f"Curated dataset {entry.id!r} produced 0 well-shaped rows "
            f"after the {entry.schema} transform. The HF source format "
            "may have changed; please file an issue."
        )
    write_jsonl(chat_rows, out_path)
    return DownloadResult(
        path=str(out_path),
        rows_loaded=rows_loaded,
        rows_kept=len(chat_rows),
    )


def _ensure_unique_ids_for_tests(entries: list[CuratedEntry]) -> None:
    """Sanity helper — exposed for tests; redundant with load_manifest's
    own duplicate detection but useful when constructing entries by hand."""
    seen: set[str] = set()
    for e in entries:
        if e.id in seen:
            raise ValueError(f"duplicate id {e.id!r}")
        seen.add(e.id)


# Hint for pkg-data tooling: this YAML must travel with the package.
__all__ = [
    "CuratedEntry",
    "DownloadResult",
    "download_curated",
    "find_entry",
    "load_manifest",
    "transform_rows",
]
