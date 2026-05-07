from unittest.mock import patch

from fastapi.testclient import TestClient

from llm_chain_sidecar.hardware.types import (
    Backend,
    CpuInfo,
    GpuDevice,
    HardwareReport,
)
from llm_chain_sidecar.main import app

client = TestClient(app)


def test_get_hardware():
    r = client.get("/api/hardware")
    assert r.status_code == 200
    j = r.json()
    assert "os" in j and "devices" in j


def test_hardware_devices_carry_capabilities():
    j = client.get("/api/hardware").json()
    assert j["devices"]
    for d in j["devices"]:
        assert "capabilities" in d
        assert "memory_kind" in d
        assert d["memory_kind"] in ("dedicated", "unified", "shared")


def _report_with_rocm() -> HardwareReport:
    return HardwareReport(
        os="Linux",
        os_version="6.5.0",
        cpu=CpuInfo(cores=8, name="fake"),
        system_ram_gb=32.0,
        devices=[
            GpuDevice(
                backend=Backend.ROCM,
                name="AMD Radeon RX 7900 XTX",
                vram_gb=24.0,
                memory_kind="dedicated",
                driver_version="6.0.32830",
            ),
            GpuDevice(backend=Backend.CPU, name="CPU", vram_gb=0.0, memory_kind="dedicated"),
        ],
    )


def test_get_hardware_marks_rocm_devices_unverified():
    with patch("llm_chain_sidecar.api.routes.probe_hardware", return_value=_report_with_rocm()):
        j = client.get("/api/hardware").json()
    rocm_devices = [d for d in j["devices"] if d["backend"] == "rocm"]
    assert len(rocm_devices) == 1
    rocm = rocm_devices[0]
    assert "rocm_unverified" in rocm["capabilities"]["warning_codes"]
    # Tier numbers come from the dedicated tier table, so a 24 GB AMD card
    # should advertise the same QLoRA cap as a 24 GB NVIDIA card.
    assert rocm["capabilities"]["qlora_max_params"] >= 13_000_000_000


def test_get_hardware_reports_rocm_experimental_armed_flag(monkeypatch):
    from llm_chain_sidecar.trainers.hf_rocm import EXPERIMENTAL_ENV_VAR

    # Default: not armed
    monkeypatch.delenv(EXPERIMENTAL_ENV_VAR, raising=False)
    assert client.get("/api/hardware").json()["rocm_experimental_armed"] is False

    # Armed via env var
    monkeypatch.setenv(EXPERIMENTAL_ENV_VAR, "1")
    assert client.get("/api/hardware").json()["rocm_experimental_armed"] is True


def test_get_models_default_excludes_restricted():
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    licenses = {m["license"] for m in models}
    assert licenses.issubset({"Apache-2.0", "MIT"})
    assert all(not m["restricted"] for m in models)


def test_get_models_filtered_by_max_params():
    r = client.get("/api/models?max_params=500000000")
    assert all(m["params"] <= 500_000_000 for m in r.json()["models"])


def test_get_models_include_restricted_returns_restricted_entries():
    r = client.get("/api/models?include_restricted=1")
    assert r.status_code == 200
    models = r.json()["models"]
    restricted = [m for m in models if m["restricted"]]
    assert restricted
    families = {m["family"] for m in restricted}
    assert {"Llama", "Gemma", "DeepSeek"}.issubset(families)
    assert all(m["license_caveat"] for m in restricted)


def test_get_models_modalities_filter_returns_only_vlms():
    r = client.get("/api/models?modalities=image")
    assert r.status_code == 200
    models = r.json()["models"]
    assert models
    assert all("image" in m["modalities"] for m in models)


def test_get_models_chat_capable_filter_hides_base_models():
    r = client.get("/api/models?chat_capable=1")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["models"]}
    assert "EleutherAI/pythia-70m" not in ids
    assert "mistralai/Mistral-7B-v0.3" not in ids
    assert "Qwen/Qwen3-1.7B" in ids


def test_get_models_modalities_filter_handles_csv():
    # text,image — entries must include both modalities.
    r = client.get("/api/models?modalities=text,image")
    assert r.status_code == 200
    models = r.json()["models"]
    assert models
    for m in models:
        assert "text" in m["modalities"]
        assert "image" in m["modalities"]


def test_create_run_returns_id_and_lists():
    body = {
        "model_id": "Qwen/Qwen3-0.6B",
        "backend": "cuda",
        "technique": "lora",
        "dataset_path": "/tmp/x.jsonl",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200
    run_id = r.json()["id"]
    listing = client.get("/api/runs").json()["runs"]
    assert any(rn["id"] == run_id for rn in listing)


def test_create_run_rejects_base_model_with_chat_dataset():
    body = {
        "model_id": "EleutherAI/pythia-70m",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": "/tmp/x.jsonl",
        "dataset_format": "jsonl_chat",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "base model" in detail
    assert "chat template" in detail
    # Suggests at least one chat-capable alternative.
    assert "Instruct" in detail or "Chat" in detail or "Qwen" in detail


def test_create_run_allows_base_model_with_csv_dataset():
    body = {
        "model_id": "EleutherAI/pythia-70m",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": "/tmp/x.csv",
        "dataset_format": "csv",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200


def test_create_run_allows_chat_capable_model_with_chat_dataset():
    body = {
        "model_id": "Qwen/Qwen3-1.7B",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": "/tmp/x.jsonl",
        "dataset_format": "jsonl_chat",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200


def test_create_run_allows_unknown_model_id_through():
    # Custom HF ids the user pastes shouldn't be blocked just because they're
    # not in the curated allowlist — only registered base entries are gated.
    body = {
        "model_id": "user/custom-fork",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": "/tmp/x.jsonl",
        "dataset_format": "jsonl_chat",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200


def test_cancel_run_returns_409_when_not_active():
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    # Run was created but never streamed, so no in-flight cancel event exists.
    r = client.post(f"/api/runs/{run_id}/cancel")
    assert r.status_code == 409


def test_get_events_returns_404_for_unknown_run():
    r = client.get("/api/runs/does-not-exist/events")
    assert r.status_code == 404


def test_get_events_returns_empty_list_when_run_has_not_streamed():
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    r = client.get(f"/api/runs/{run_id}/events")
    assert r.status_code == 200
    assert r.json() == {"events": []}


def test_get_events_replays_persisted_events(tmp_path):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.executor import _append_event
    from llm_chain_sidecar.trainers.base import EventType, TrainingEvent

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    run = routes_mod._store.get(run_id)

    _append_event(run.output_dir, TrainingEvent(type=EventType.START, message="x"))
    _append_event(run.output_dir, TrainingEvent(type=EventType.STEP, step=1, total_steps=2, loss=2.0))
    _append_event(run.output_dir, TrainingEvent(type=EventType.STEP, step=2, total_steps=2, loss=1.5))
    _append_event(run.output_dir, TrainingEvent(type=EventType.DONE))

    r = client.get(f"/api/runs/{run_id}/events")
    assert r.status_code == 200
    events = r.json()["events"]
    assert [e["type"] for e in events] == ["start", "step", "step", "done"]
    assert [e["loss"] for e in events if e["type"] == "step"] == [2.0, 1.5]


def test_export_gguf_404s_for_unknown_run():
    r = client.post("/api/runs/does-not-exist/export/gguf")
    assert r.status_code == 404


def test_export_gguf_404s_when_run_not_succeeded():
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    # Fresh run is "pending" — export must reject until it succeeds.
    r = client.post(f"/api/runs/{run_id}/export/gguf")
    assert r.status_code == 404


def test_export_gguf_rejects_unknown_quant():
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    r = client.post(f"/api/runs/{run_id}/export/gguf?quant=zzz")
    assert r.status_code == 400


def test_export_gguf_starts_export_for_succeeded_run(monkeypatch):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    # Don't actually shell out to peft/llama.cpp; intercept at the worker.
    monkeypatch.setattr(routes_mod, "_run_gguf_export", lambda *_a, **_kw: None)

    r = client.post(f"/api/runs/{run_id}/export/gguf?quant=q8_0")
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "running"
    assert body["quant"] == "q8_0"


def test_get_export_gguf_404_when_no_export_started():
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    r = client.get(f"/api/runs/{run_id}/export/gguf")
    assert r.status_code == 404


def test_get_export_gguf_returns_stored_state(tmp_path):
    from llm_chain_sidecar.api import routes as routes_mod

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._write_gguf_state(
        run_id, {"status": "done", "path": "/x/y.gguf", "quant": "q4_k_m"}
    )
    r = client.get(f"/api/runs/{run_id}/export/gguf")
    assert r.status_code == 200
    assert r.json()["status"] == "done"
    assert r.json()["path"] == "/x/y.gguf"


def test_export_hub_404s_for_unknown_run():
    r = client.post("/api/runs/does-not-exist/export/hub", json={"repo_id": "u/r"})
    assert r.status_code == 404


def test_export_hub_404s_when_run_not_succeeded():
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    r = client.post(f"/api/runs/{run_id}/export/hub", json={"repo_id": "u/r"})
    assert r.status_code == 404


def test_export_hub_returns_401_when_not_signed_in(monkeypatch):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.exports import hub as hub_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    monkeypatch.setattr(hub_mod, "_resolve_token", lambda: None)
    r = client.post(f"/api/runs/{run_id}/export/hub", json={"repo_id": "u/r"})
    assert r.status_code == 401
    assert "Not signed in" in r.json()["detail"]


def test_export_hub_succeeds_when_signed_in(monkeypatch):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.exports import hub as hub_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    captured = {}
    def fake_push(rid, repo_id, runs_root, private, folder):
        captured.update(rid=rid, repo_id=repo_id, private=private, folder=folder)
        return f"https://huggingface.co/{repo_id}"

    monkeypatch.setattr(hub_mod, "_resolve_token", lambda: "hf_xxx")
    monkeypatch.setattr(routes_mod.exports, "push_to_hub", fake_push)

    r = client.post(
        f"/api/runs/{run_id}/export/hub",
        json={"repo_id": "user/my-adapter", "private": False, "folder": "adapter"},
    )
    assert r.status_code == 200
    assert r.json()["url"] == "https://huggingface.co/user/my-adapter"
    assert captured == {
        "rid": run_id,
        "repo_id": "user/my-adapter",
        "private": False,
        "folder": "adapter",
    }


def test_get_hf_auth_status_reflects_resolver(monkeypatch):
    from llm_chain_sidecar.exports import hub as hub_mod

    monkeypatch.setattr(hub_mod, "_resolve_token", lambda: None)
    assert client.get("/api/auth/hf").json() == {"signed_in": False}

    monkeypatch.setattr(hub_mod, "_resolve_token", lambda: "hf_xxx")
    assert client.get("/api/auth/hf").json() == {"signed_in": True}
