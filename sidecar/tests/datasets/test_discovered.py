import json
from pathlib import Path

import pytest

from llm_chain_sidecar.datasets.discovered import (
    DiscoveredEntry,
    default_watched_dir,
    list_discovered,
)


def test_default_watched_dir_honours_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LLM_CHAIN_WATCHED_DATASETS_DIR", str(tmp_path))
    assert default_watched_dir() == tmp_path


def test_list_discovered_returns_empty_when_missing(tmp_path: Path):
    assert list_discovered(tmp_path / "ghost") == []


def test_list_discovered_returns_files(tmp_path: Path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "yo"},
                ]
            }
        )
        + "\n"
    )
    (tmp_path / "b.csv").write_text("text\nrow1\nrow2\n")
    (tmp_path / "notes.txt").write_text("ignored")
    entries = list_discovered(tmp_path)
    names = [e.name for e in entries]
    assert "a.jsonl" in names
    assert "b.csv" in names
    # .txt is unrecognised → format_hint is None; still listed so user can pick.
    assert "notes.txt" in names


def test_list_discovered_sets_format_hints(tmp_path: Path):
    (tmp_path / "x.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n"
    )
    (tmp_path / "y.csv").write_text("text\nrow\n")
    (tmp_path / "z.tsv").write_text("col1\tcol2\nrow1\trow2\n")
    by_name = {e.name: e for e in list_discovered(tmp_path)}
    assert by_name["x.jsonl"].format_hint == "jsonl_chat"
    assert by_name["y.csv"].format_hint == "csv"
    # TSV intentionally returns no format hint — the trainer's CSV
    # loader doesn't auto-detect tab delimiters, so a discovered .tsv
    # routes through the workshop where the user picks the delimiter
    # explicitly.
    assert by_name["z.tsv"].format_hint is None


def test_list_discovered_detects_dpo_format_via_first_row(tmp_path: Path):
    """Regression: .jsonl files used to default to jsonl_chat even
    when the rows were DPO-shaped — the user only learned at training
    time. We now peek at row 1 to disambiguate."""
    (tmp_path / "dpo.jsonl").write_text(
        json.dumps(
            {
                "prompt": "what is 2+2?",
                "chosen": "4",
                "rejected": "five",
            }
        )
        + "\n"
    )
    [entry] = list_discovered(tmp_path)
    assert entry.format_hint == "jsonl_dpo"


def test_list_discovered_caps_jsonl_row_count(tmp_path: Path):
    """Counting a 100M-row file synchronously would block the route.
    Cap at 100k rows and surface a ``row_count_capped`` flag."""
    from llm_chain_sidecar.datasets import discovered as d_mod

    p = tmp_path / "big.jsonl"
    # Cheap: write more rows than the cap by stamping them all out.
    body = (
        json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n"
    )
    p.write_text(body * (d_mod._MAX_COUNT_ROWS + 5))
    [entry] = list_discovered(tmp_path)
    assert entry.row_count == d_mod._MAX_COUNT_ROWS
    assert entry.row_count_capped is True


def test_list_discovered_does_not_skip_underscore_prefix(tmp_path: Path):
    """Regression: ``_`` is a legitimate ML naming idiom (``_eval.jsonl``,
    ``_v2.jsonl``) and the previous skip prefix hid these files
    silently."""
    (tmp_path / "_eval.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n"
    )
    [entry] = list_discovered(tmp_path)
    assert entry.name == "_eval.jsonl"


def test_list_discovered_rejects_symlinks_outside_root(tmp_path: Path):
    """Defence-in-depth: a symlink inside the watched folder
    pointing outside it would let the trainer read arbitrary files
    when the picker forwards the path. Resolve and refuse."""
    import os

    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.jsonl"
    secret.write_text("{}\n")
    inside = tmp_path / "inside"
    inside.mkdir()
    try:
        os.symlink(secret, inside / "linked.jsonl")
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this platform")
    entries = list_discovered(inside)
    names = [e.name for e in entries]
    assert "linked.jsonl" not in names


def test_list_discovered_handles_permission_error(tmp_path: Path, monkeypatch):
    """macOS Sonoma+ TCC can lock down ~/Documents. Surface as
    empty list (which the UI renders with the same amber hint) rather
    than letting the route 500."""
    def boom(self):
        raise PermissionError("TCC denied")

    monkeypatch.setattr(Path, "iterdir", boom)
    assert list_discovered(tmp_path) == []


def test_list_discovered_counts_jsonl_rows(tmp_path: Path):
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "a"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "b"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "c"}]}) + "\n"
    )
    [entry] = list_discovered(tmp_path)
    assert entry.row_count == 3
    assert entry.error is None


def test_list_discovered_skips_blank_lines_in_jsonl_count(tmp_path: Path):
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"messages": []}) + "\n\n\n"
        + json.dumps({"messages": []}) + "\n"
    )
    [entry] = list_discovered(tmp_path)
    assert entry.row_count == 2


def test_list_discovered_surfaces_non_utf8_error(tmp_path: Path):
    """Non-UTF-8 .jsonl content surfaces an error on the entry. The
    format peek now happens before the row count, so a non-UTF-8
    file falls back to the default jsonl_chat hint (peek returns
    None) and the row counter records the UTF-8 problem under
    ``error``."""
    (tmp_path / "broken.jsonl").write_bytes(b"\xff\xfe\xfa")
    [entry] = list_discovered(tmp_path)
    assert entry.error is not None
    assert "UTF-8" in entry.error


def test_list_discovered_skips_dotfiles_and_os_scratch(tmp_path: Path):
    """Only OS-scratch (``.foo``, ``~foo``, named scratch files) get
    filtered. ``_foo.jsonl`` is intentionally kept — it's a
    legitimate ML naming convention."""
    (tmp_path / "real.jsonl").write_text("{}\n")
    (tmp_path / ".DS_Store").write_bytes(b"")
    (tmp_path / ".hidden.jsonl").write_text("{}\n")
    (tmp_path / "Thumbs.db").write_bytes(b"")
    names = sorted(e.name for e in list_discovered(tmp_path))
    assert names == ["real.jsonl"]


def test_list_discovered_does_not_recurse(tmp_path: Path):
    """One level deep — nested files belong to the regular picker."""
    (tmp_path / "top.jsonl").write_text("{}\n")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "deep.jsonl").write_text("{}\n")
    names = [e.name for e in list_discovered(tmp_path)]
    assert names == ["top.jsonl"]


def test_list_discovered_sorts_by_mtime_descending(tmp_path: Path):
    """Latest drop should appear first in the picker."""
    import os
    import time

    p1 = tmp_path / "old.jsonl"
    p2 = tmp_path / "new.jsonl"
    p1.write_text("{}\n")
    p2.write_text("{}\n")
    # Force distinct mtimes — filesystems with low resolution would
    # otherwise tie and the sort would be insertion-order.
    os.utime(p1, (time.time() - 100, time.time() - 100))
    os.utime(p2, (time.time(), time.time()))
    names = [e.name for e in list_discovered(tmp_path)]
    assert names == ["new.jsonl", "old.jsonl"]
