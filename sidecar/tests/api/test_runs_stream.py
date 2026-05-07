from unittest.mock import patch

from fastapi.testclient import TestClient

from llm_chain_sidecar.main import app
from llm_chain_sidecar.trainers.base import EventType, TrainingEvent


def test_stream_run_emits_sse_events():
    fake_events = [
        TrainingEvent(type=EventType.START),
        TrainingEvent(type=EventType.STEP, step=1, total_steps=1, loss=1.0),
        TrainingEvent(type=EventType.DONE),
    ]
    client = TestClient(app)
    body = {"model_id": "m", "backend": "cuda", "technique": "lora",
            "dataset_path": "/tmp/x", "epochs": 1}
    run_id = client.post("/api/runs", json=body).json()["id"]
    with patch("llm_chain_sidecar.api.routes._executor.execute",
               return_value=iter(fake_events)):
        with client.stream("GET", f"/api/runs/{run_id}/stream") as r:
            body_text = "".join(r.iter_text())
    assert "event: start" in body_text
    assert "event: step" in body_text
    assert "event: done" in body_text
    assert '"loss":1.0' in body_text
