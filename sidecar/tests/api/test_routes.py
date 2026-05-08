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


def test_export_hub_kicks_off_async_push_and_polls_to_done(monkeypatch, existing_jsonl_chat):
    """Hub push is now async (mirrors GGUF): POST returns 202 immediately,
    a background thread does the upload, GET reads the latest state from
    disk. Pre-fix the route blocked synchronously and the UI showed a
    frozen 'Pushing…' spinner for the whole upload."""
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
    def fake_push(rid, repo_id, runs_root, private, folder, on_progress=None):
        captured.update(rid=rid, repo_id=repo_id, private=private, folder=folder)
        return f"https://huggingface.co/{repo_id}"

    monkeypatch.setattr(hub_mod, "_resolve_token", lambda: "hf_xxx")
    monkeypatch.setattr(routes_mod.exports, "push_to_hub", fake_push)

    r = client.post(
        f"/api/runs/{run_id}/export/hub",
        json={"repo_id": "user/my-adapter", "private": False, "folder": "adapter"},
    )
    assert r.status_code == 202
    initial = r.json()
    assert initial["status"] == "running"
    assert initial["repo_id"] == "user/my-adapter"

    # Wait briefly for the background worker (it's a no-op fake_push,
    # so completes near-instantly) and check the GET endpoint surfaces
    # the terminal state.
    import time
    deadline = time.monotonic() + 2.0
    state = None
    while time.monotonic() < deadline:
        state = client.get(f"/api/runs/{run_id}/export/hub").json()
        if state.get("status") == "done":
            break
        time.sleep(0.05)
    assert state is not None and state.get("status") == "done"
    assert state["url"] == "https://huggingface.co/user/my-adapter"
    assert captured == {
        "rid": run_id,
        "repo_id": "user/my-adapter",
        "private": False,
        "folder": "adapter",
    }


def test_get_hub_export_returns_404_when_no_push_started(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    r = client.get(f"/api/runs/{run_id}/export/hub")
    assert r.status_code == 404


def test_export_hub_writes_failed_state_when_push_raises(monkeypatch, existing_jsonl_chat):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.exports import hub as hub_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    def fake_push(rid, repo_id, runs_root, private, folder, on_progress=None):
        raise RuntimeError("network died")

    monkeypatch.setattr(hub_mod, "_resolve_token", lambda: "hf_xxx")
    monkeypatch.setattr(routes_mod.exports, "push_to_hub", fake_push)

    r = client.post(
        f"/api/runs/{run_id}/export/hub",
        json={"repo_id": "user/repo"},
    )
    assert r.status_code == 202

    import time
    deadline = time.monotonic() + 2.0
    state = None
    while time.monotonic() < deadline:
        state = client.get(f"/api/runs/{run_id}/export/hub").json()
        if state.get("status") == "failed":
            break
        time.sleep(0.05)
    assert state is not None and state.get("status") == "failed"
    assert "network died" in state.get("error", "")


def test_preview_dataset_returns_first_n_rows(existing_jsonl_chat):
    """The dataset picker uses this so users can confirm their data
    parses correctly without clicking Train and waiting for an mlx_lm
    crash. Returns the loader's actual output, capped at ``limit``."""
    r = client.post(
        "/api/datasets/preview",
        json={
            "dataset_path": existing_jsonl_chat,
            "dataset_format": "jsonl_chat",
            "limit": 3,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["row_count"] == 1
    assert body["shown"] == 1
    assert "messages" in body["rows"][0]


def test_preview_dataset_caps_limit():
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    fd = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    p = _Path(fd.name)
    fd.close()
    p.write_text(
        "\n".join(
            _json.dumps({"messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]})
            for i in range(20)
        ) + "\n"
    )
    r = client.post(
        "/api/datasets/preview",
        json={"dataset_path": str(p), "dataset_format": "jsonl_chat", "limit": 5},
    )
    assert r.status_code == 200
    assert r.json()["shown"] == 5
    assert r.json()["row_count"] == 20


def test_preview_dataset_surfaces_loader_errors_as_400(tmp_path):
    p = tmp_path / "broken.jsonl"
    p.write_text('{"not_messages": []}\n')
    r = client.post(
        "/api/datasets/preview",
        json={"dataset_path": str(p), "dataset_format": "jsonl_chat"},
    )
    assert r.status_code == 400
    assert "missing 'messages'" in r.json()["detail"]


def test_preview_dataset_rejects_missing_path():
    r = client.post(
        "/api/datasets/preview",
        json={
            "dataset_path": "/tmp/never-exists.jsonl",
            "dataset_format": "jsonl_chat",
        },
    )
    assert r.status_code == 400


def test_delete_run_removes_run_dir_for_terminal_states(existing_jsonl_chat):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    r = client.delete(f"/api/runs/{run_id}")
    assert r.status_code == 200
    assert r.json() == {"deleted": True}
    # Subsequent GET should 404.
    assert client.get(f"/api/runs/{run_id}").status_code == 404


def test_delete_run_refuses_in_flight_run(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    # Fresh run is PENDING — delete must refuse with 409 to keep the
    # trainer from finding its dir gone mid-step.
    r = client.delete(f"/api/runs/{run_id}")
    assert r.status_code == 409
    assert "Cancel it first" in r.json()["detail"]


def test_delete_run_404s_for_unknown_id():
    assert client.delete("/api/runs/notanid").status_code == 404
    assert client.delete("/api/runs/abcdef012345").status_code == 404


def test_count_dataset_returns_jsonl_row_count(tmp_path):
    """The Train page hits this on every dataset change. Must be cheap
    (no row parsing) and return the right number — pinning the contract
    so a future loader change doesn't accidentally regress to slow."""
    import json as _json

    p = tmp_path / "x.jsonl"
    rows = [
        _json.dumps({"messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]})
        for i in range(7)
    ]
    p.write_text("\n".join(rows) + "\n\n\n")  # trailing blank lines must not count
    r = client.post(
        "/api/datasets/count",
        json={"dataset_path": str(p), "dataset_format": "jsonl_chat"},
    )
    assert r.status_code == 200
    assert r.json() == {"row_count": 7, "format": "jsonl_chat"}


def test_count_dataset_csv_excludes_header(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("text\nrow1\nrow2\nrow3\n")
    r = client.post(
        "/api/datasets/count",
        json={
            "dataset_path": str(p),
            "dataset_format": "csv",
            "text_column": "text",
        },
    )
    assert r.status_code == 200
    assert r.json()["row_count"] == 3


def test_count_dataset_text_dir_recurses(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("b")
    (sub / "c.txt").write_text("c")
    r = client.post(
        "/api/datasets/count",
        json={"dataset_path": str(tmp_path), "dataset_format": "text_dir"},
    )
    assert r.status_code == 200
    assert r.json()["row_count"] == 3


def test_count_dataset_hf_hub_returns_count_from_card_metadata(monkeypatch):
    """Most HF dataset cards publish split-level row counts in their
    auto-generated dataset_info block; reading that gives us an exact
    number without downloading anything. Pin the integration via a
    fake HfApi.dataset_info."""
    from llm_chain_sidecar.api import routes as routes_mod

    monkeypatch.setattr(
        routes_mod,
        "_hf_hub_train_split_count",
        lambda repo_id: 12345 if repo_id == "acme/dataset" else None,
    )
    r = client.post(
        "/api/datasets/count",
        json={"dataset_path": "acme/dataset", "dataset_format": "hf_hub"},
    )
    assert r.status_code == 200
    assert r.json() == {"row_count": 12345, "format": "hf_hub"}


def test_count_dataset_hf_hub_falls_back_to_null_when_metadata_missing(monkeypatch):
    """For older datasets that don't publish dataset_info in the card,
    return null and let the UI show 'count unavailable, will be checked
    at training time'."""
    from llm_chain_sidecar.api import routes as routes_mod

    monkeypatch.setattr(routes_mod, "_hf_hub_train_split_count", lambda repo_id: None)
    r = client.post(
        "/api/datasets/count",
        json={"dataset_path": "acme/old-dataset", "dataset_format": "hf_hub"},
    )
    assert r.status_code == 200
    assert r.json() == {"row_count": None, "format": "hf_hub"}


def test_count_dataset_hf_hub_rejects_empty_repo_id():
    r = client.post(
        "/api/datasets/count",
        json={"dataset_path": "  ", "dataset_format": "hf_hub"},
    )
    assert r.status_code == 400


def test_hf_hub_train_split_count_walks_card_data_shape():
    """Direct unit test of the metadata extractor: HF returns
    card_data sometimes as a dict, sometimes a typed object, and
    dataset_info can be a list of configs or a single dict. The
    helper tolerates both shapes and returns the first 'train' split's
    num_examples."""
    from llm_chain_sidecar.api.routes import _hf_hub_train_split_count
    from unittest.mock import patch, MagicMock

    fake_info = MagicMock()
    fake_info.card_data = {
        "dataset_info": {
            "splits": [
                {"name": "train", "num_examples": 9876},
                {"name": "validation", "num_examples": 100},
            ]
        }
    }
    with patch("huggingface_hub.HfApi") as HfApi:
        HfApi.return_value.dataset_info.return_value = fake_info
        assert _hf_hub_train_split_count("acme/x") == 9876


def test_hf_hub_train_split_count_handles_dict_splits_shape():
    from llm_chain_sidecar.api.routes import _hf_hub_train_split_count
    from unittest.mock import patch, MagicMock

    # Some configs lay out splits as a dict keyed by name rather than
    # as a list. Shouldn't matter to the caller.
    fake_info = MagicMock()
    fake_info.card_data = {
        "dataset_info": {
            "splits": {
                "train": {"name": "train", "num_examples": 200},
                "test": {"name": "test", "num_examples": 50},
            }
        }
    }
    with patch("huggingface_hub.HfApi") as HfApi:
        HfApi.return_value.dataset_info.return_value = fake_info
        assert _hf_hub_train_split_count("acme/x") == 200


def test_hf_hub_train_split_count_returns_none_on_network_error():
    from llm_chain_sidecar.api.routes import _hf_hub_train_split_count
    from unittest.mock import patch

    with patch("huggingface_hub.HfApi") as HfApi:
        HfApi.return_value.dataset_info.side_effect = RuntimeError("network down")
        assert _hf_hub_train_split_count("acme/x") is None


def test_generate_endpoint_emits_status_frames_for_cache_misses(monkeypatch, existing_jsonl_chat):
    """When the playground cache is empty / has a different run, the
    inference module emits a 'status' frame before the first token so
    the UI shows 'Loading model…' instead of staring at a spinner.
    Pin the SSE wire format."""
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.inference import GenerationToken
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    def fake_stream(run_dict, cfg, runs_root, cancel_event=None):
        yield GenerationToken(status="Loading model into memory…")
        yield GenerationToken(text="hi")
        yield GenerationToken(done=True)

    monkeypatch.setattr(routes_mod, "generate_stream", fake_stream)
    with client.stream(
        "POST",
        f"/api/runs/{run_id}/generate",
        json={"prompt": "hello"},
    ) as r:
        text = "".join(r.iter_text())
    assert "event: status" in text
    assert "Loading model" in text
    assert "event: token" in text
    assert "event: done" in text


def test_generate_endpoint_passes_cancel_event_to_inference(monkeypatch, existing_jsonl_chat):
    """The route's generator must forward a fresh cancel_event into
    generate_stream so the HF backend's StoppingCriteria has something
    to poll. The actual disconnect-detected → cancel_event.set() chain
    is timing-sensitive over TestClient (Starlette's StreamingResponse
    only notices the disconnect when it next tries to send), so this
    test pins the wiring contract: the inference module receives a
    real threading.Event it can check.
    """
    import threading as _threading
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.inference import GenerationToken
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    captured: dict[str, object] = {}

    def fake_stream(run_dict, cfg, runs_root, cancel_event=None):
        captured["cancel_event"] = cancel_event
        yield GenerationToken(text="ok")
        yield GenerationToken(done=True)

    monkeypatch.setattr(routes_mod, "generate_stream", fake_stream)
    with client.stream(
        "POST",
        f"/api/runs/{run_id}/generate",
        json={"prompt": "hi"},
    ) as r:
        list(r.iter_text())  # drain
    assert isinstance(captured["cancel_event"], _threading.Event)
    # The cancel_event reaches inference unset; it's the route's
    # finally (or the HF stream's own finally on consumer abandon)
    # that flips it. Verifying both halves end-to-end requires a
    # disconnect we can drive deterministically; the unit-test
    # equivalent lives in test_inference_playground.


def test_count_dataset_csv_requires_text_column(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("text\nrow1\n")
    r = client.post(
        "/api/datasets/count",
        json={"dataset_path": str(p), "dataset_format": "csv"},
    )
    assert r.status_code == 400


def test_resume_run_creates_child_with_parent_config(monkeypatch, existing_jsonl_chat):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    parent_body = {
        "model_id": "Qwen/Qwen3-0.6B",
        "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat",
        "epochs": 1, "lora_rank": 8, "lora_alpha": 16,
    }
    parent_id = client.post("/api/runs", json=parent_body).json()["id"]
    routes_mod._store.update_status(parent_id, RunStatus.SUCCEEDED)

    r = client.post(
        f"/api/runs/{parent_id}/resume",
        json={"epochs": 2, "learning_rate": 1e-4},
    )
    assert r.status_code == 200
    child_id = r.json()["id"]
    child = routes_mod._store.get(child_id)
    # Inherits parent's model + backend + LoRA shape; overrides epochs + lr.
    assert child.config.model_id == parent_body["model_id"]
    assert child.config.backend == parent_body["backend"]
    assert child.config.lora_rank == parent_body["lora_rank"]
    assert child.config.lora_alpha == parent_body["lora_alpha"]
    assert child.config.epochs == 2
    assert child.config.learning_rate == 1e-4
    assert child.config.resume_from == parent_id


def test_resume_run_404s_when_parent_not_succeeded(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    parent_id = client.post("/api/runs", json=body).json()["id"]
    # Parent is PENDING.
    r = client.post(f"/api/runs/{parent_id}/resume", json={"epochs": 2})
    assert r.status_code == 404


def test_resume_run_rejects_zero_epochs(monkeypatch, existing_jsonl_chat):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    parent_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(parent_id, RunStatus.SUCCEEDED)
    r = client.post(f"/api/runs/{parent_id}/resume", json={"epochs": 0})
    assert r.status_code == 400


def test_create_run_with_resume_from_must_match_lora_shape(monkeypatch, existing_jsonl_chat):
    """LoRA shapes don't match across rank/alpha changes — resuming
    with a different shape would either crash at adapter load or
    silently corrupt the new training. Reject at the API."""
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    parent_body = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
        "lora_rank": 8, "lora_alpha": 16,
    }
    parent_id = client.post("/api/runs", json=parent_body).json()["id"]
    routes_mod._store.update_status(parent_id, RunStatus.SUCCEEDED)

    child_body = {
        **parent_body,
        "lora_rank": 16,  # mismatch
        "resume_from": parent_id,
    }
    r = client.post("/api/runs", json=child_body)
    assert r.status_code == 400
    assert "rank/alpha" in r.json()["detail"]


def test_generate_endpoint_refuses_vlm_runs(monkeypatch, existing_jsonl_chat):
    """Playground only handles text-in/text-out. VLM runs need
    image inputs + Vision2Seq model class — refuse upfront."""
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "mlx_vlm", "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat_vision",
        "epochs": 1,
    }
    # Bypass route validation that requires a real vision dataset; we
    # just need a SUCCEEDED row to test the gate.
    run_id = routes_mod._store.create(routes_mod.RunConfig(**body)).id
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)
    r = client.post(
        f"/api/runs/{run_id}/generate",
        json={"prompt": "hi"},
    )
    assert r.status_code == 400
    assert "vision-language" in r.json()["detail"]


def test_generate_endpoint_streams_tokens(monkeypatch, existing_jsonl_chat):
    """Smoke test for the SSE wire format. Mocks generate_stream so we
    don't need a real model loaded."""
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.inference import GenerationToken
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    def fake_stream(run_dict, cfg, runs_root, cancel_event=None):
        yield GenerationToken(text="Hello")
        yield GenerationToken(text=", ")
        yield GenerationToken(text="world!")
        yield GenerationToken(done=True)

    monkeypatch.setattr(routes_mod, "generate_stream", fake_stream)
    with client.stream(
        "POST",
        f"/api/runs/{run_id}/generate",
        json={"prompt": "hi"},
    ) as r:
        text = "".join(r.iter_text())
    assert "event: token" in text
    assert '"text": "Hello"' in text
    assert "event: done" in text


def test_delete_run_frees_inference_cache(monkeypatch, existing_jsonl_chat):
    """When a SUCCEEDED run gets deleted, any cached model the
    playground had warm for that run must be dropped — otherwise its
    tensors stay in RAM with no on-disk run to back them."""
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar import inference as _inference
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    monkeypatch.setattr(_inference, "cached_run_id", lambda: run_id)
    freed = []
    monkeypatch.setattr(_inference, "free_cache", lambda: freed.append(True))

    r = client.delete(f"/api/runs/{run_id}")
    assert r.status_code == 200
    assert freed == [True]


def test_list_runs_includes_adapter_size_for_succeeded_runs(monkeypatch, existing_jsonl_chat):
    """The Library page reads adapter_size_bytes off /api/runs to show
    on-disk size at a glance. Sum of every adapter file under the run
    dir, including HF Trainer checkpoint subdirs."""
    from pathlib import Path as _Path

    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    run = routes_mod._store.get(run_id)
    # Drop a fake adapter file in the run dir.
    adapter = _Path(run.output_dir) / "adapter_model.safetensors"
    adapter.write_bytes(b"\x00" * 12345)

    listing = client.get("/api/runs").json()["runs"]
    row = next(r for r in listing if r["id"] == run_id)
    assert row["adapter_size_bytes"] == 12345


def test_list_runs_omits_adapter_size_for_non_succeeded(existing_jsonl_chat):
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    listing = client.get("/api/runs").json()["runs"]
    row = next(r for r in listing if r["id"] == run_id)
    # PENDING runs don't get the field — their on-disk state is
    # transient and would just confuse users.
    assert "adapter_size_bytes" not in row or row["adapter_size_bytes"] is None


def test_cleanup_endpoint_deletes_only_matching_terminal_runs(monkeypatch, existing_jsonl_chat):
    """The cleanup sweep takes a status filter + age cutoff. Pending
    and running runs are NEVER deletable through this endpoint, no
    matter what statuses are requested."""
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    failed_id = client.post("/api/runs", json=body).json()["id"]
    canceled_id = client.post("/api/runs", json=body).json()["id"]
    pending_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(failed_id, RunStatus.FAILED)
    routes_mod._store.update_status(canceled_id, RunStatus.CANCELED)
    # pending_id stays PENDING

    r = client.post(
        "/api/maintenance/cleanup",
        json={"older_than_days": 0, "statuses": ["failed", "canceled"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body["deleted_ids"]) == {failed_id, canceled_id}
    # Pending run survives.
    listing = {row["id"] for row in client.get("/api/runs").json()["runs"]}
    assert pending_id in listing
    assert failed_id not in listing


def test_cleanup_endpoint_rejects_unknown_status():
    r = client.post(
        "/api/maintenance/cleanup",
        json={"older_than_days": 0, "statuses": ["pending"]},
    )
    assert r.status_code == 400
    assert "pending" in r.json()["detail"]


def test_cleanup_endpoint_respects_age_cutoff(monkeypatch, existing_jsonl_chat):
    """A 7-day cutoff shouldn't delete runs created today."""
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.FAILED)

    r = client.post(
        "/api/maintenance/cleanup",
        json={"older_than_days": 7, "statuses": ["failed"]},
    )
    assert r.status_code == 200
    assert run_id not in r.json()["deleted_ids"]
    assert client.get(f"/api/runs/{run_id}").status_code == 200


def test_lr_finder_creates_three_runs_with_different_lrs(monkeypatch, existing_jsonl_chat):
    """The Train-page button spawns one short run per LR. Each
    inherits the user's full RunConfig but overrides learning_rate +
    max_steps + purpose."""
    from llm_chain_sidecar.api import routes as routes_mod

    cfg = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat",
        "epochs": 1, "learning_rate": 2e-4,
    }
    # Stub the background runner so the test doesn't try to actually
    # spawn mlx_lm — we just want to verify the runs were created.
    monkeypatch.setattr(
        routes_mod._executor, "execute", lambda rid: iter([]),
    )

    r = client.post(
        "/api/runs/lr-finder",
        json={
            "config": cfg,
            "learning_rates": [1e-4, 2e-4, 5e-4],
            "steps_per_run": 10,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["run_ids"]) == 3
    runs = [routes_mod._store.get(rid) for rid in body["run_ids"]]
    seen_lrs = sorted(r.config.learning_rate for r in runs)
    assert seen_lrs == [1e-4, 2e-4, 5e-4]
    assert all(r.config.max_steps == 10 for r in runs)
    assert all(r.config.purpose == "lr_finder" for r in runs)
    # resume_from must be cleared even if the user's config carried
    # one — each sniff starts from random init.
    assert all(r.config.resume_from is None for r in runs)


def test_lr_finder_rejects_negative_lr(existing_jsonl_chat):
    cfg = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat", "epochs": 1,
    }
    r = client.post(
        "/api/runs/lr-finder",
        json={"config": cfg, "learning_rates": [-0.0001, 2e-4]},
    )
    assert r.status_code == 400


def test_lr_finder_rejects_too_many_lrs(existing_jsonl_chat):
    cfg = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat", "epochs": 1,
    }
    r = client.post(
        "/api/runs/lr-finder",
        json={"config": cfg, "learning_rates": [1e-4] * 7},
    )
    assert r.status_code == 400
    assert "capped at 6" in r.json()["detail"]


def test_lr_finder_rolls_back_partial_creates_on_validation_failure(monkeypatch, existing_jsonl_chat):
    """If LR #2 fails validation after LR #1 was already persisted,
    the orphan run #1 must be deleted so the Runs page doesn't
    accumulate noise from failed sweeps."""
    from fastapi import HTTPException as _HTTPException

    from llm_chain_sidecar.api import routes as routes_mod

    cfg = {
        "model_id": "m", "backend": "mlx", "technique": "lora",
        "dataset_path": existing_jsonl_chat,
        "dataset_format": "jsonl_chat", "epochs": 1,
    }
    before = {row["id"] for row in client.get("/api/runs").json()["runs"]}

    real_validate = routes_mod._validate_run_config
    call_count = [0]

    def flaky_validate(c):
        call_count[0] += 1
        if call_count[0] == 2:
            raise _HTTPException(status_code=400, detail="synthetic validation failure")
        real_validate(c)

    monkeypatch.setattr(routes_mod, "_validate_run_config", flaky_validate)
    monkeypatch.setattr(
        routes_mod._executor, "execute", lambda rid: iter([]),
    )

    r = client.post(
        "/api/runs/lr-finder",
        json={"config": cfg, "learning_rates": [1e-4, 2e-4, 5e-4]},
    )
    assert r.status_code == 400
    after = {row["id"] for row in client.get("/api/runs").json()["runs"]}
    # No new runs — the first one was rolled back when the second failed.
    assert after == before


def test_eval_endpoint_streams_role_indexed_tokens(monkeypatch, existing_jsonl_chat):
    """Eval suite SSE: each token frame is tagged (role, prompt_index, text)
    so the UI can fill the right cell of the side-by-side table without
    extra server state."""
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.inference.eval_suite import EvalFrame
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    def fake_evaluate(run_dict, cfg, runs_root, cancel_event=None, skip_event=None):
        yield EvalFrame(status="Loading base…")
        yield EvalFrame(role="base", prompt_index=0, text="Base says hi.")
        yield EvalFrame(role="adapter", prompt_index=0, text="Adapter says hi.")
        yield EvalFrame(done=True)

    monkeypatch.setattr(routes_mod, "evaluate", fake_evaluate)
    with client.stream(
        "POST",
        f"/api/runs/{run_id}/eval",
        json={"prompts": ["hi"]},
    ) as r:
        text = "".join(r.iter_text())
    assert "event: status" in text
    assert "Loading base" in text
    assert '"role": "base"' in text
    assert '"role": "adapter"' in text
    assert '"prompt_index": 0' in text
    assert "event: done" in text


def test_skip_eval_prompt_409s_when_no_eval_running():
    """Stale Skip clicks after the suite finishes shouldn't 500 — they
    return 409 so the UI can treat them as no-ops without an error
    banner."""
    r = client.post("/api/runs/abcdef012345/eval/skip")
    assert r.status_code == 409


def test_skip_eval_prompt_signals_running_eval(monkeypatch, existing_jsonl_chat):
    """While an eval suite is in flight, POST /eval/skip flips the
    per-run skip flag the orchestrator polls. Mock the eval generator
    so we can assert the signal landed without driving a real model."""
    import threading as _threading
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.inference.eval_suite import EvalFrame
    from llm_chain_sidecar.runs.types import RunStatus

    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": existing_jsonl_chat, "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    seen_skip = _threading.Event()
    started = _threading.Event()

    def fake_evaluate(run_dict, cfg, runs_root, cancel_event=None, skip_event=None):
        started.set()
        # Block on the skip signal so the test can deterministically
        # observe the flow: skip POST → flag set → generator
        # observes → moves on.
        if skip_event is not None and skip_event.wait(timeout=2):
            seen_skip.set()
        yield EvalFrame(done=True)

    monkeypatch.setattr(routes_mod, "evaluate", fake_evaluate)

    # Open the eval stream in a background thread so we can fire
    # the skip POST while it's running.
    def consume_stream():
        with client.stream("POST", f"/api/runs/{run_id}/eval", json={"prompts": ["x"]}) as r:
            for _ in r.iter_text():
                pass

    t = _threading.Thread(target=consume_stream, daemon=True)
    t.start()
    assert started.wait(timeout=2), "fake_evaluate never entered"

    r = client.post(f"/api/runs/{run_id}/eval/skip")
    assert r.status_code == 200
    assert r.json() == {"signaled": True}
    assert seen_skip.wait(timeout=2), "skip flag was never observed by orchestrator"
    t.join(timeout=2)


def test_eval_endpoint_refuses_vlm_runs(monkeypatch, existing_jsonl_chat):
    from llm_chain_sidecar.api import routes as routes_mod
    from llm_chain_sidecar.runs.types import RunStatus

    cfg = routes_mod.RunConfig(
        model_id="m", backend="mlx_vlm", technique="lora",
        dataset_path=existing_jsonl_chat,
        dataset_format="jsonl_chat_vision", epochs=1,
    )
    run_id = routes_mod._store.create(cfg).id
    routes_mod._store.update_status(run_id, RunStatus.SUCCEEDED)

    r = client.post(
        f"/api/runs/{run_id}/eval",
        json={"prompts": ["hi"]},
    )
    assert r.status_code == 400


def test_get_hf_auth_status_reflects_resolver(monkeypatch):
    from llm_chain_sidecar.exports import hub as hub_mod

    monkeypatch.setattr(hub_mod, "_resolve_token", lambda: None)
    assert client.get("/api/auth/hf").json() == {"signed_in": False}

    monkeypatch.setattr(hub_mod, "_resolve_token", lambda: "hf_xxx")
    assert client.get("/api/auth/hf").json() == {"signed_in": True}


# ---------------------------------------------------------------------------
# Dataset workshop (F-A1)
# ---------------------------------------------------------------------------


def test_build_dataset_writes_jsonl_chat_and_returns_stats(tmp_path):
    out = tmp_path / "out.jsonl"
    r = client.post(
        "/api/datasets/build",
        json={
            "raw_text": "user,assistant\nhi,hello\nbye,goodbye\n",
            "input_format": "csv",
            "target": "chat",
            "user_field": "user",
            "assistant_field": "assistant",
            "output_path": str(out),
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["path"] == str(out)
    assert body["stats"]["output_rows"] == 2
    assert out.exists()

    # And the result must round-trip through the JSONL chat loader the
    # trainer actually uses — guards against shipping a workshop that
    # writes "valid-looking" JSONL the loader rejects.
    from llm_chain_sidecar.datasets.loader import load_dataset
    from llm_chain_sidecar.datasets.types import DatasetFormat, DatasetSource

    loaded = load_dataset(
        DatasetSource(format=DatasetFormat.JSONL_CHAT, path=str(out))
    )
    assert len(loaded) == 2
    assert loaded[0]["messages"][0]["role"] == "user"


def test_build_dataset_reports_dropped_duplicates(tmp_path):
    out = tmp_path / "dedup.jsonl"
    r = client.post(
        "/api/datasets/build",
        json={
            "raw_text": "u,a\nx,y\nx,y\np,q\n",
            "input_format": "csv",
            "target": "chat",
            "user_field": "u",
            "assistant_field": "a",
            "dedupe": True,
            "output_path": str(out),
        },
    )
    assert r.status_code == 200, r.text
    stats = r.json()["stats"]
    assert stats["output_rows"] == 2
    assert stats["dropped_duplicate"] == 1


def test_build_dataset_rejects_when_both_text_and_path_given(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("u,a\nx,y\np,q\n")
    out = tmp_path / "x.jsonl"
    r = client.post(
        "/api/datasets/build",
        json={
            "raw_text": "u,a\n",
            "source_path": str(src),
            "input_format": "csv",
            "target": "chat",
            "user_field": "u",
            "assistant_field": "a",
            "output_path": str(out),
        },
    )
    assert r.status_code == 400
    assert "exactly one" in r.json()["detail"]


def test_build_dataset_reads_from_source_path(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("u,a\nhi,hello\nbye,goodbye\n")
    out = tmp_path / "out.jsonl"
    r = client.post(
        "/api/datasets/build",
        json={
            "source_path": str(src),
            "input_format": "csv",
            "target": "chat",
            "user_field": "u",
            "assistant_field": "a",
            "output_path": str(out),
        },
    )
    assert r.status_code == 200
    assert out.exists()


def test_build_dataset_refuses_to_overwrite(tmp_path):
    out = tmp_path / "exists.jsonl"
    out.write_text("placeholder\n")
    r = client.post(
        "/api/datasets/build",
        json={
            "raw_text": "u,a\nx,y\np,q\n",
            "input_format": "csv",
            "target": "chat",
            "user_field": "u",
            "assistant_field": "a",
            "output_path": str(out),
        },
    )
    assert r.status_code == 409


def test_build_dataset_rejects_when_all_rows_get_dropped(tmp_path):
    """Cleaning toggles can leave nothing behind. The endpoint should
    400 with an actionable hint instead of writing an empty file the
    trainer would later reject for being too small."""
    out = tmp_path / "x.jsonl"
    r = client.post(
        "/api/datasets/build",
        json={
            "raw_text": "u,a\n,\n,\n",  # both fields empty on every row
            "input_format": "csv",
            "target": "chat",
            "user_field": "u",
            "assistant_field": "a",
            "output_path": str(out),
        },
    )
    assert r.status_code == 400


def test_build_dataset_rejects_single_surviving_row(tmp_path):
    """Trainer needs ≥2 rows for the train/val split; surface here so
    the user fixes the dataset before the trainer's ValueError fires."""
    out = tmp_path / "x.jsonl"
    r = client.post(
        "/api/datasets/build",
        json={
            "raw_text": "u,a\nhi,hello\n",
            "input_format": "csv",
            "target": "chat",
            "user_field": "u",
            "assistant_field": "a",
            "output_path": str(out),
        },
    )
    assert r.status_code == 400
    assert "at least 2" in r.json()["detail"]


def test_synth_rejects_when_neither_source_provided():
    r = client.post(
        "/api/datasets/synth",
        json={"topic": "cooking", "style": "friendly", "count": 1},
    )
    assert r.status_code == 400
    assert "exactly one source" in r.json()["detail"]


def test_synth_rejects_unknown_backend():
    r = client.post(
        "/api/datasets/synth",
        json={
            "source_model_id": "Qwen/Qwen3-1.7B",
            "source_backend": "made-up",
            "topic": "x",
            "count": 1,
        },
    )
    assert r.status_code == 400
    assert "Unknown source_backend" in r.json()["detail"]


def test_synth_rejects_unknown_model_id():
    r = client.post(
        "/api/datasets/synth",
        json={
            "source_model_id": "ghost/no-such-model",
            "source_backend": "mlx",
            "topic": "x",
            "count": 1,
        },
    )
    assert r.status_code == 400
    assert "Unknown source_model_id" in r.json()["detail"]


def test_synth_rejects_vlm_backend():
    r = client.post(
        "/api/datasets/synth",
        json={
            "source_model_id": "Qwen/Qwen3-1.7B",
            "source_backend": "mlx_vlm",
            "topic": "x",
            "count": 1,
        },
    )
    assert r.status_code == 400


def test_synth_streams_rows_with_stubbed_collector(monkeypatch):
    """End-to-end SSE shape check: with the playground stream stubbed,
    the route should emit one row event per generated row plus a
    terminating done event."""
    import json as _json

    from llm_chain_sidecar.inference import synth as _synth

    def _fake_collect(run_dict, gcfg, runs_root, cancel_event):
        return _json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )

    monkeypatch.setattr(_synth, "_collect_stream", _fake_collect)

    # Need a real chat-capable model id from the registry. Pull the
    # first chat-capable text-only one so the test isn't pinned to a
    # specific model that registry churn could remove.
    from llm_chain_sidecar.api import routes as routes_mod

    chat_entries = [
        e
        for e in routes_mod._registry.entries(include_restricted=True)
        if e.chat_capable and "image" not in e.modalities
    ]
    assert chat_entries, "registry must ship at least one chat-capable model"
    model_id = chat_entries[0].id

    with client.stream(
        "POST",
        "/api/datasets/synth",
        json={
            "source_model_id": model_id,
            "source_backend": "cpu",
            "topic": "test",
            "style": "friendly",
            "count": 2,
        },
    ) as resp:
        assert resp.status_code == 200
        events = list(resp.iter_lines())

    # Concatenate and split into SSE frames.
    body = "\n".join(events)
    assert "event: row" in body
    assert body.count("event: row") == 2
    assert "event: done" in body


def test_build_dataset_passthrough_jsonl_chat(tmp_path):
    """When the user pastes already-chat-shaped JSONL, the workshop
    should run the cleaners on it and write the survivors back without
    asking for user/assistant column names."""
    out = tmp_path / "passthrough.jsonl"
    import json as _json

    text = "\n".join(
        _json.dumps({"messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]})
        for i in range(3)
    ) + "\n"
    r = client.post(
        "/api/datasets/build",
        json={
            "raw_text": text,
            "input_format": "jsonl",
            "passthrough_chat": True,
            "output_path": str(out),
        },
    )
    assert r.status_code == 200
    assert r.json()["stats"]["output_rows"] == 3
