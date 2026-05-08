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


def test_create_run_returns_id_and_lists(existing_jsonl_chat):
    body = {
        "model_id": "Qwen/Qwen3-0.6B",
        "backend": "cuda",
        "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200
    run_id = r.json()["id"]
    listing = client.get("/api/runs").json()["runs"]
    assert any(rn["id"] == run_id for rn in listing)


def test_create_run_rejects_base_model_with_chat_dataset(existing_jsonl_chat):
    body = {
        "model_id": "EleutherAI/pythia-70m",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": existing_jsonl_chat,
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


def test_create_run_allows_base_model_with_csv_dataset(existing_csv):
    body = {
        "model_id": "EleutherAI/pythia-70m",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": existing_csv,
        "dataset_format": "csv",
        "text_column": "text",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200


def test_create_run_allows_chat_capable_model_with_chat_dataset(existing_jsonl_chat):
    body = {
        "model_id": "Qwen/Qwen3-1.7B",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200


def test_create_run_allows_unknown_model_id_through(existing_jsonl_chat):
    # Custom HF ids the user pastes shouldn't be blocked just because they're
    # not in the curated allowlist — only registered base entries are gated.
    body = {
        "model_id": "user/custom-fork",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200


def test_create_run_rejects_unknown_dataset_format(existing_jsonl_chat):
    body = {
        "model_id": "Qwen/Qwen3-0.6B",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "weird_format",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "Unknown dataset_format" in r.json()["detail"]


def test_create_run_rejects_unknown_backend(existing_jsonl_chat):
    body = {
        "model_id": "m",
        "backend": "tpu",  # not in _KNOWN_BACKENDS
        "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "Unknown backend" in r.json()["detail"]


def test_create_run_rejects_unknown_technique(existing_jsonl_chat):
    body = {
        "model_id": "m",
        "backend": "mlx",
        "technique": "magic",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "Unknown technique" in r.json()["detail"]


def test_create_run_rejects_zero_epochs(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "dataset_format": "jsonl_chat",
        "epochs": 0,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "epochs" in r.json()["detail"]


def test_create_run_rejects_negative_batch_size(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "dataset_format": "jsonl_chat",
        "batch_size": -1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400


def test_create_run_rejects_nonpositive_learning_rate(existing_jsonl_chat):
    """A zero or negative learning rate silently produces a flat / runaway
    loss curve. Reject at the boundary so the user gets a clear hint
    instead of debugging a no-op fine-tune.

    NaN/Inf are also rejected at runtime via math.isfinite, but strict JSON
    can't even encode them — clients cannot reach that branch. Negative
    finite values are the realistic boundary case.
    """
    body = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "dataset_format": "jsonl_chat",
        "learning_rate": -0.0001,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "learning_rate" in r.json()["detail"]


def test_create_run_rejects_oversized_lora_rank(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "dataset_format": "jsonl_chat",
        "lora_rank": 99999,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "lora_rank" in r.json()["detail"]


def test_create_run_rejects_directory_for_jsonl_format(tmp_path):
    """text_dir wants a directory; everything else wants a file. Submitting
    a folder where a JSONL was expected used to error mid-staging with an
    IsADirectoryError that read like a Python bug."""
    body = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": str(tmp_path),  # directory, not a file
        "dataset_format": "jsonl_chat",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "must be a file" in r.json()["detail"]


def test_create_run_rejects_file_for_text_dir_format(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat,  # a file, not a dir
        "dataset_format": "text_dir",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "must be a directory" in r.json()["detail"]


def test_create_run_rejects_empty_hf_hub_id():
    body = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": "  ",  # whitespace-only
        "dataset_format": "hf_hub",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "dataset id" in r.json()["detail"]


def test_create_run_rejects_missing_dataset_path():
    body = {
        "model_id": "Qwen/Qwen3-0.6B",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": "/tmp/definitely-does-not-exist.jsonl",
        "dataset_format": "jsonl_chat",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "does not exist" in r.json()["detail"]


def test_create_run_rejects_csv_without_text_column(existing_csv):
    body = {
        "model_id": "Qwen/Qwen3-0.6B",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": existing_csv,
        "dataset_format": "csv",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "text column" in r.json()["detail"].lower()


def test_create_run_rejects_vision_dataset_on_non_vlm_backend(existing_jsonl_chat):
    body = {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "backend": "mlx",  # text backend, not mlx_vlm
        "technique": "lora",
        "dataset_path": existing_jsonl_chat,  # path content shape doesn't matter — backend check trips first
        "dataset_format": "jsonl_chat_vision",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "VLM backend" in r.json()["detail"]


def test_create_run_rejects_vlm_backend_on_non_vision_dataset(existing_jsonl_chat):
    body = {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "backend": "mlx_vlm",
        "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "jsonl_chat_vision" in r.json()["detail"]


def test_create_run_rejects_vision_dataset_on_text_only_model(existing_jsonl_chat):
    body = {
        "model_id": "Qwen/Qwen3-0.6B",  # text-only registry entry
        "backend": "mlx_vlm",
        "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat_vision",
        "epochs": 1,
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 400
    assert "text-only" in r.json()["detail"]


def test_cancel_run_returns_409_when_not_active(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    # Run was created but never streamed, so no in-flight cancel event exists.
    r = client.post(f"/api/runs/{run_id}/cancel")
    assert r.status_code == 409


def test_get_events_returns_404_for_unknown_run():
    r = client.get("/api/runs/does-not-exist/events")
    assert r.status_code == 404


def test_get_run_rejects_path_traversal_run_id():
    """run_id from the URL path lands in a filesystem path inside the
    storage layer. A 12-hex-char check at the route boundary keeps an
    attacker from probing for run.json files outside runs_root via
    `/api/runs/..%2Fetc/run.json` style URLs."""
    # FastAPI URL-decodes %2F into a path separator that the routing
    # layer would split on, but the request reaches us as a single
    # path param when the client encodes it as a literal slash. The
    # cheaper attack vector is just sending dot-segments.
    r = client.get("/api/runs/..%2F..%2Fetc/events")
    # FastAPI returns 404 for unmatched paths even before our handler.
    # Either 404 or our handler's 404 is fine — the contract is "no
    # data leak", not "exact status code".
    assert r.status_code == 404

    # A direct route hit with a malformed but URL-safe id has to be 404
    # from our validator (not a 500 from a path stat error).
    r = client.get("/api/runs/not-12-hex-chars/events")
    assert r.status_code == 404
    r = client.get("/api/runs/zzzzzzzzzzzz/events")  # 12 chars but not hex
    assert r.status_code == 404


def test_cancel_run_rejects_invalid_run_id():
    r = client.post("/api/runs/notanid/cancel")
    assert r.status_code == 404


def test_export_gguf_rejects_invalid_run_id():
    r = client.post("/api/runs/notanid/export/gguf")
    assert r.status_code == 404
    r = client.get("/api/runs/notanid/export/gguf")
    assert r.status_code == 404


def test_create_run_rejects_oversized_model_id(existing_jsonl_chat):
    """A 100KB model_id used to be accepted, written to run.json, and only
    fail at HF download time. RunConfig now bounds it to 512 chars at the
    Pydantic boundary so the storage layer never sees runaway strings."""
    body = {
        "model_id": "a" * 1024,
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat",
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 422  # Pydantic ValidationError → FastAPI 422


def test_create_run_rejects_oversized_dataset_path(existing_jsonl_chat):
    body = {
        "model_id": "m",
        "backend": "mlx",
        "technique": "lora",
        "dataset_path": "/tmp/" + ("x" * 10_000),
        "dataset_format": "jsonl_chat",
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 422


def test_get_events_returns_empty_list_when_run_has_not_streamed(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    r = client.get(f"/api/runs/{run_id}/events")
    assert r.status_code == 200
    assert r.json() == {"events": []}


def test_get_events_replays_persisted_events(tmp_path, existing_jsonl_chat):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.executor import _append_event
    from llm_chain_sidecar.trainers.base import EventType, TrainingEvent

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
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


def test_export_gguf_404s_when_run_not_succeeded(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    # Fresh run is "pending" — export must reject until it succeeds.
    r = client.post(f"/api/runs/{run_id}/export/gguf")
    assert r.status_code == 404


def test_export_gguf_rejects_unknown_quant(existing_jsonl_chat):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    r = client.post(f"/api/runs/{run_id}/export/gguf?quant=zzz")
    assert r.status_code == 400


def test_export_gguf_starts_export_for_succeeded_run(monkeypatch, existing_jsonl_chat):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
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


def test_get_export_gguf_404_when_no_export_started(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    r = client.get(f"/api/runs/{run_id}/export/gguf")
    assert r.status_code == 404


def test_write_gguf_state_is_atomic_under_concurrent_reads(tmp_path, existing_jsonl_chat):
    """Hammer the state file from multiple threads; readers must never see
    truncated JSON. Pre-fix, _write_gguf_state used Path.write_text which
    truncates the file before re-writing — a concurrent _read_gguf_state
    midway through that window read empty bytes and returned None even
    though the export was actively running."""
    import threading

    from llm_chain_sidecar.api import routes as routes_mod

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]

    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            i += 1
            routes_mod._write_gguf_state(run_id, {"status": "running", "tick": i})

    torn_reads = []

    def reader():
        for _ in range(200):
            s = routes_mod._read_gguf_state(run_id)
            # state is None iff the file genuinely doesn't exist OR a
            # torn write happened. Once writer() has run once the file
            # exists, so any None read after that is a torn write.
            if s is None:
                torn_reads.append(True)

    w = threading.Thread(target=writer, daemon=True)
    r = threading.Thread(target=reader, daemon=True)
    w.start()
    # Make sure writer has produced the file at least once before reader starts.
    while routes_mod._read_gguf_state(run_id) is None:
        pass
    r.start()
    r.join(timeout=2)
    stop.set()
    w.join(timeout=2)

    assert torn_reads == [], f"saw {len(torn_reads)} torn reads — state writes aren't atomic"


def test_get_export_gguf_returns_stored_state(tmp_path, existing_jsonl_chat):
    from llm_chain_sidecar.api import routes as routes_mod

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
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


def test_export_hub_404s_when_run_not_succeeded(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    r = client.post(f"/api/runs/{run_id}/export/hub", json={"repo_id": "u/r"})
    assert r.status_code == 404


def test_export_hub_returns_401_when_not_signed_in(monkeypatch, existing_jsonl_chat):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.exports import hub as hub_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    monkeypatch.setattr(hub_mod, "_resolve_token", lambda: None)
    r = client.post(f"/api/runs/{run_id}/export/hub", json={"repo_id": "u/r"})
    assert r.status_code == 401
    assert "Not signed in" in r.json()["detail"]


def test_export_hub_succeeds_when_signed_in(monkeypatch, existing_jsonl_chat):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.exports import hub as hub_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
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
