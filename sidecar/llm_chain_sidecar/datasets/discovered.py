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
_SKIP_PREFIXES: tuple[str, ...] = (".", "~", "_")
_SKIP_NAMES = {"Thumbs.db", "desktop.ini", ".DS_Store"}


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

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "modified_unix": self.modified_unix,
            "format_hint": self.format_hint,
            "row_count": self.row_count,
            "error": self.error,
        }


def _guess_format(path: Path) -> str | None:
    """Map an extension to a DatasetFormat string. Returns None for
    extensions we don't recognise — the UI surfaces these as "unknown
    format" and offers them through the regular picker anyway."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        # Default to plain JSONL chat. The DPO format also lives in
        # .jsonl files; the user can override on the picker. We don't
        # peek at the file's first row to disambiguate because that's
        # the kind of guessing that bites later.
        return DatasetFormat.JSONL_CHAT.value
    if suffix == ".csv":
        return DatasetFormat.CSV.value
    if suffix == ".tsv":
        # CSV-like; the user picks a delimiter on the workshop. The
        # picker treats this the same as .csv.
        return DatasetFormat.CSV.value
    return None


def _count_jsonl_rows(path: Path) -> tuple[int | None, str | None]:
    """Cheap line-count for JSONL files. Returns (count, error).

    Uses the same "count non-empty lines" approach as the
    /api/datasets/count endpoint — we don't parse rows here, just
    shape an at-a-glance number for the picker's badge.
    """
    try:
        count = 0
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count, None
    except UnicodeDecodeError as e:
        return None, f"not valid UTF-8 (offset {e.start})"
    except OSError as e:
        return None, f"read failed: {e}"


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
    entries: list[DiscoveredEntry] = []
    for child in root.iterdir():
        if not child.is_file():
            continue
        if child.name in _SKIP_NAMES:
            continue
        if child.name.startswith(_SKIP_PREFIXES):
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        fmt = _guess_format(child)
        row_count: int | None = None
        err: str | None = None
        if fmt and child.suffix.lower() == ".jsonl":
            row_count, err = _count_jsonl_rows(child)
        entries.append(
            DiscoveredEntry(
                path=str(child),
                name=child.name,
                size_bytes=stat.st_size,
                modified_unix=stat.st_mtime,
                format_hint=fmt,
                row_count=row_count,
                error=err,
            )
        )
    entries.sort(key=lambda e: e.modified_unix, reverse=True)
    return entries


__all__ = [
    "DiscoveredEntry",
    "default_watched_dir",
    "list_discovered",
]
