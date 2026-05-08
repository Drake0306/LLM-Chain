from pathlib import Path

import pytest

from llm_chain_sidecar.runs.store import RunStore
from llm_chain_sidecar.runs.types import RunConfig, RunStatus


def test_create_run_persists_config(tmp_path: Path):
    store = RunStore(root=tmp_path)
    cfg = RunConfig(
        model_id="Qwen/Qwen3-0.6B",
        backend="cuda",
        technique="lora",
        dataset_path="/tmp/x.jsonl",
        epochs=1,
    )
    run = store.create(cfg)
    assert run.id
    assert run.status == RunStatus.PENDING
    loaded = store.get(run.id)
    assert loaded.config.model_id == "Qwen/Qwen3-0.6B"


def test_list_runs_returns_newest_first(tmp_path: Path):
    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    a = store.create(cfg)
    b = store.create(cfg)
    runs = store.list()
    assert runs[0].id == b.id
    assert runs[1].id == a.id


def test_update_status(tmp_path: Path):
    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    store.update_status(run.id, RunStatus.RUNNING)
    assert store.get(run.id).status == RunStatus.RUNNING


def test_list_caches_until_invalidated(tmp_path: Path):
    """The Runs page hits list() on every navigation. Without caching,
    each call reads N run.json files from disk, which stalls noticeably
    for users with hundreds of runs. The cache must invalidate when a
    run is created, status-updated, or deleted so the UI never serves
    stale data after a write."""
    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)

    # First call populates the cache.
    a = store.create(cfg)
    initial = store.list()
    assert len(initial) == 1

    # A second list() should return the same cached object identity —
    # cheap proxy for "didn't re-read disk".
    cached = store.list()
    assert cached is initial

    # create() must invalidate.
    b = store.create(cfg)
    after_create = store.list()
    assert after_create is not initial
    assert {r.id for r in after_create} == {a.id, b.id}

    # update_status() must invalidate too — otherwise the Runs page
    # would show stale 'pending' badges after the executor finished.
    store.update_status(a.id, RunStatus.SUCCEEDED)
    fresh = store.list()
    assert next(r for r in fresh if r.id == a.id).status == RunStatus.SUCCEEDED


def test_delete_removes_dir_and_invalidates_cache(tmp_path: Path):
    store = RunStore(root=tmp_path)
    cfg = RunConfig(model_id="m", backend="cuda", technique="lora",
                    dataset_path="/tmp/x", epochs=1)
    run = store.create(cfg)
    store.list()  # warm cache

    store.delete(run.id)
    assert not (tmp_path / run.id).exists()
    # Cache must reflect the deletion.
    assert store.list() == []
    # get() on a deleted run raises FileNotFoundError, which the route
    # layer converts to 404.
    with pytest.raises(FileNotFoundError):
        store.get(run.id)


def test_delete_unknown_id_raises(tmp_path: Path):
    store = RunStore(root=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.delete("does-not-exist")
