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
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "bye"}]}) + "\n"
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
        json.dumps({"messages": []}) + "\n"
    )
    (tmp_path / "y.csv").write_text("text\nrow\n")
    (tmp_path / "z.tsv").write_text("col1\tcol2\nrow1\trow2\n")
    by_name = {e.name: e for e in list_discovered(tmp_path)}
    assert by_name["x.jsonl"].format_hint == "jsonl_chat"
    assert by_name["y.csv"].format_hint == "csv"
    # TSV maps to CSV format hint — the workshop handles delimiter
    # detection; the picker uses the same code path either way.
    assert by_name["z.tsv"].format_hint == "csv"


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
    (tmp_path / "broken.jsonl").write_bytes(b"\xff\xfe\xfa")
    [entry] = list_discovered(tmp_path)
    assert entry.error is not None
    assert "UTF-8" in entry.error


def test_list_discovered_skips_dotfiles_and_os_scratch(tmp_path: Path):
    (tmp_path / "real.jsonl").write_text("{}\n")
    (tmp_path / ".DS_Store").write_bytes(b"")
    (tmp_path / ".hidden.jsonl").write_text("{}\n")
    (tmp_path / "Thumbs.db").write_bytes(b"")
    (tmp_path / "_temp.jsonl").write_text("{}\n")
    names = [e.name for e in list_discovered(tmp_path)]
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
