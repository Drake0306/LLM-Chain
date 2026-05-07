from pathlib import Path

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
