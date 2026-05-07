from fastapi.testclient import TestClient

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


def test_cancel_run_returns_409_when_not_active():
    body = {
        "model_id": "m", "backend": "cuda", "technique": "lora",
        "dataset_path": "/tmp/x.jsonl", "epochs": 1,
    }
    run_id = client.post("/api/runs", json=body).json()["id"]
    # Run was created but never streamed, so no in-flight cancel event exists.
    r = client.post(f"/api/runs/{run_id}/cancel")
    assert r.status_code == 409
