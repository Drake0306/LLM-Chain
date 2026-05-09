"""Discovered datasets (F-D15).

Lists files dropped into a watched folder so the user doesn't have
to click through the file picker every time. The folder defaults
to ``~/Documents/llm-chain-datasets/`` and can be overridden via
``LLM_CHAIN_WATCHED_DATASETS_DIR`` (mostly for tests; in prod the
desktop's Settings UI writes the same env var to the launcher
config).

Implementation: fresh-on-demand directory scan rather than a
``watchdog`` background observer. Pros: no new dep, no thread
management, no platform-specific FS-event handling. Cons: not
real-time. The Dataset picker calls /api/datasets/discovered when
the user opens the section, which reads the folder fresh — a
hundred-file scan is sub-millisecond on every platform we ship.

Format guessing: each file's extension picks the format hint we
surface in the UI. The hint is advisory — the user can still
pick the file via the regular picker and override the format.
``.jsonl`` files get a row count via the same fast counter the
preview endpoint uses; we don't parse the rows here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .types import DatasetFormat

# Skip OS scratch / hidden files when listing the watched folder.
# ``_`` is intentionally NOT in the skip list — ML projects routinely
# name files ``_eval.jsonl`` or ``_v2.jsonl`` and silently hiding them
# is the worse default. Only OS-scratch (``.``, ``~``) gets filtered.
_SKIP_PREFIXES: tuple[str, ...] = (".", "~")
_SKIP_NAMES = {"Thumbs.db", "desktop.ini", ".DS_Store"}

# Cap on JSONL row counting so a 100M-row file doesn't block the
# route worker for tens of seconds. The picker shows
# "{N}+ rows (capped)" when the count hit the cap.
_MAX_COUNT_ROWS = 100_000


def default_watched_dir() -> Path:
    """Resolve the watched folder, honouring the test override."""
    env = os.environ.get("LLM_CHAIN_WATCHED_DATASETS_DIR")
    if env:
        return Path(env)
    return Path.home() / "Documents" / "llm-chain-datasets"


@dataclass
class DiscoveredEntry:
    """One file the discovered scan returned."""

    path: str
    name: str
    size_bytes: int
    modified_unix: float
    format_hint: str | None
    row_count: int | None
    error: str | None = None
    # True when the JSONL row counter hit ``_MAX_COUNT_ROWS`` without
    # reading the entire file. The UI shows the badge as
    # ``{count}+ rows (capped)`` so the user knows the number is a
    # lower bound, not the actual size.
    row_count_capped: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "modified_unix": self.modified_unix,
            "format_hint": self.format_hint,
            "row_count": self.row_count,
            "row_count_capped": self.row_count_capped,
            "error": self.error,
        }


def _peek_jsonl_format(path: Path) -> str | None:
    """Look at the first non-empty row to disambiguate jsonl_chat
    vs jsonl_dpo. Cheap — we only read a few KB.

    Returns ``jsonl_dpo`` when the row carries the prompt/chosen/
    rejected triple, ``jsonl_chat`` for messages-shaped rows, and
    None when the row doesn't fit either (the picker treats null as
    "unknown format" and the user picks explicitly).

    Read errors (binary content, permission denied) return None so
    the caller falls back to the default jsonl_chat hint and the
    row counter handles surfacing the underlying error.
    """
    import json as _json

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except _json.JSONDecodeError:
                    return None
                if not isinstance(obj, dict):
                    return None
                if (
                    "prompt" in obj
                    and "chosen" in obj
                    and "rejected" in obj
                ):
                    return DatasetFormat.JSONL_DPO.value
                if "messages" in obj:
                    return DatasetFormat.JSONL_CHAT.value
                return None
    except (OSError, UnicodeDecodeError):
        return None
    return None


def _guess_format(path: Path) -> str | None:
    """Map an extension to a DatasetFormat string. Returns None for
    extensions we don't recognise — the UI surfaces these as "unknown
    format" and offers them through the regular picker anyway.

    For .jsonl we also peek at the first row so DPO files don't
    silently pick up the chat format hint — the user would only
    notice at training time when the trainer rejects the row shape.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _peek_jsonl_format(path) or DatasetFormat.JSONL_CHAT.value
    if suffix == ".csv":
        return DatasetFormat.CSV.value
    if suffix == ".tsv":
        # The trainer's CSV loader doesn't auto-detect tab delimiters,
        # so we leave .tsv unmapped; the user routes through the
        # Workshop where the delimiter is explicit. Returning None
        # here makes the picker disable the file with a "use the
        # workshop" hint rather than enabling it for a format that
        # would fail downstream.
        return None
    return None


def _count_jsonl_rows(
    path: Path,
    cap: int = _MAX_COUNT_ROWS,
) -> tuple[int | None, bool, str | None]:
    """Cheap line-count for JSONL files. Returns (count, capped, error).

    Caps at ``_MAX_COUNT_ROWS`` so a 100M-row file doesn't block the
    route worker for seconds. The capped flag lets the UI show
    "{N}+" instead of pretending the count is exact.
    """
    try:
        count = 0
        capped = False
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    count += 1
                    if count >= cap:
                        # Stop counting; mark the result as a lower bound.
                        capped = True
                        break
        return count, capped, None
    except UnicodeDecodeError as e:
        return None, False, f"not valid UTF-8 (offset {e.start})"
    except OSError as e:
        return None, False, f"read failed: {e}"


def list_discovered(watched_dir: Path | None = None) -> list[DiscoveredEntry]:
    """Snapshot the watched folder. Returns entries sorted by mtime
    descending so the user's latest drop sits at the top.

    Recursion is intentionally one level deep: we list files in the
    watched dir itself, ignoring subdirectories. A nested layout
    blurs the "drop a file here" mental model and the picker would
    then have to tree-render. If the user wants nested datasets,
    the regular file picker handles them.
    """
    root = (watched_dir or default_watched_dir())
    if not root.exists() or not root.is_dir():
        return []
    try:
        children = list(root.iterdir())
    except PermissionError:
        # macOS Sonoma+ TCC can lock down ~/Documents until the user
        # grants Full Disk Access. Surface as "no entries" rather
        # than 500ing the route — the picker's amber hint covers
        # the missing-folder case and PermissionError lands there too.
        return []
    except OSError:
        return []
    resolved_root = root.resolve()
    entries: list[DiscoveredEntry] = []
    for child in children:
        if not child.is_file():
            continue
        if child.name in _SKIP_NAMES:
            continue
        if child.name.startswith(_SKIP_PREFIXES):
            continue
        # Symlink containment: a symlink inside the watched folder
        # could point outside it. The trainer would then read
        # arbitrary user files when the picker forwards the path.
        # Resolve the child and refuse if it escapes.
        try:
            resolved_child = child.resolve()
            resolved_child.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        fmt = _guess_format(child)
        row_count: int | None = None
        capped = False
        err: str | None = None
        if fmt and child.suffix.lower() == ".jsonl":
            row_count, capped, err = _count_jsonl_rows(child)
        entries.append(
            DiscoveredEntry(
                path=str(child),
                name=child.name,
                size_bytes=stat.st_size,
                modified_unix=stat.st_mtime,
                format_hint=fmt,
                row_count=row_count,
                row_count_capped=capped,
                error=err,
            )
        )
    # Tiebreaker on name so two files with the same mtime (FAT32 /
    # network FS / same-second writes) sort deterministically across
    # calls — the UI relies on stable order between refreshes.
    entries.sort(key=lambda e: (-e.modified_unix, e.name))
    return entries


__all__ = [
    "DiscoveredEntry",
    "default_watched_dir",
    "list_discovered",
]
