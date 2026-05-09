import json
from pathlib import Path

import pytest

from llm_chain_sidecar.exports import leaderboard


class _FakeResp:
    def __init__(self, status_code: int = 200, body: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


def test_build_payload_includes_required_fields():
    payload = leaderboard.build_payload(
        run_id="r1",
        repo_id="user/my-adapter",
        base_model="acme/base",
    )
    assert payload["repo_id"] == "user/my-adapter"
    assert payload["base_model"] == "acme/base"
    assert payload["run_id"] == "r1"
    assert "app_version" in payload
    # Optional fields stay out when not provided.
    assert "eval_results" not in payload
    assert "notes" not in payload


def test_build_payload_includes_optional_fields_when_set():
    payload = leaderboard.build_payload(
        run_id="r1",
        repo_id="user/x",
        base_model="b",
        eval_results={"loss": 0.42},
        notes="trained on emoji-heavy chat",
        license_name="MIT",
    )
    assert payload["eval_results"] == {"loss": 0.42}
    assert payload["notes"] == "trained on emoji-heavy chat"
    assert payload["license"] == "MIT"


def test_build_payload_strips_whitespace_and_omits_blanks():
    payload = leaderboard.build_payload(
        run_id="r",
        repo_id="  user/x  ",
        base_model="  acme/base  ",
        notes="   ",
        license_name="",
    )
    assert payload["repo_id"] == "user/x"
    assert payload["base_model"] == "acme/base"
    assert "notes" not in payload
    assert "license" not in payload


def test_build_payload_rejects_empty_repo_id():
    with pytest.raises(ValueError, match="repo_id"):
        leaderboard.build_payload(run_id="r", repo_id="   ", base_model="b")


def test_submit_raises_when_endpoint_unconfigured(monkeypatch):
    monkeypatch.delenv("LLM_CHAIN_LEADERBOARD_URL", raising=False)
    with pytest.raises(leaderboard.LeaderboardNotConfiguredError):
        leaderboard.submit({"repo_id": "x"})


def test_submit_uses_env_endpoint_when_no_override(monkeypatch):
    monkeypatch.setenv(
        "LLM_CHAIN_LEADERBOARD_URL", "https://example.invalid/submit",
    )
    captured: dict = {}

    def fake_post(url, json=None, timeout=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResp(200, {"id": "queued-1"})

    result = leaderboard.submit(
        {"repo_id": "x", "base_model": "b"}, requester=fake_post,
    )
    assert captured["url"] == "https://example.invalid/submit"
    assert result.endpoint == "https://example.invalid/submit"
    assert result.response == {"id": "queued-1"}
    assert "User-Agent" in captured["headers"]


def test_submit_explicit_endpoint_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_CHAIN_LEADERBOARD_URL", "https://env/submit")
    captured: dict = {}

    def fake_post(url, json=None, timeout=None, headers=None):
        captured["url"] = url
        return _FakeResp(200, {})

    leaderboard.submit(
        {"repo_id": "x", "base_model": "b"},
        endpoint="https://override/path",
        requester=fake_post,
    )
    assert captured["url"] == "https://override/path"


def test_submit_raises_on_non_2xx(monkeypatch):
    def fake_post(url, json=None, timeout=None, headers=None):
        return _FakeResp(500, {}, text="server exploded")

    with pytest.raises(leaderboard.LeaderboardSubmissionError, match="500"):
        leaderboard.submit(
            {"repo_id": "x", "base_model": "b"},
            endpoint="https://example/x",
            requester=fake_post,
        )


def test_submit_wraps_transport_errors(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("dns down")

    with pytest.raises(leaderboard.LeaderboardSubmissionError, match="dns down"):
        leaderboard.submit(
            {"repo_id": "x", "base_model": "b"},
            endpoint="https://example/x",
            requester=boom,
        )


def test_build_payload_scrubs_sensitive_keys_from_eval_results():
    """Defence-in-depth: a UI bug or curl typo could put a local
    dataset path into eval_results. The build helper must strip
    known-sensitive keys before the payload leaves the box."""
    payload = leaderboard.build_payload(
        run_id="r",
        repo_id="user/x",
        base_model="b",
        eval_results={
            "loss": 0.42,
            "dataset_path": "/Users/roy/Documents/secret/data.jsonl",
            "Output_Dir": "/var/run/run-1",
            "meta": {
                "token": "should-not-leak",
                "score": 0.91,
            },
        },
    )
    er = payload["eval_results"]
    # Whitelisted-by-omission keys survive.
    assert er["loss"] == 0.42
    assert er["meta"]["score"] == 0.91
    # Sensitive keys (case-insensitive) get dropped.
    assert "dataset_path" not in er
    assert "Output_Dir" not in er
    # Recursion strips nested too.
    assert "token" not in er["meta"]


def test_build_payload_uses_app_version_default():
    """Default app_version pulls from recipes.APP_VERSION so the
    payload field tracks releases instead of a stale literal."""
    from llm_chain_sidecar.recipes import APP_VERSION

    payload = leaderboard.build_payload(
        run_id="r", repo_id="user/x", base_model="b",
    )
    assert payload["app_version"] == APP_VERSION


def test_submit_user_agent_uses_live_app_version(monkeypatch):
    from llm_chain_sidecar.recipes import APP_VERSION

    captured: dict = {}

    def fake_post(url, json=None, timeout=None, headers=None):
        captured["headers"] = headers
        return _FakeResp(200, {})

    leaderboard.submit(
        {"repo_id": "x", "base_model": "b"},
        endpoint="https://example/x",
        requester=fake_post,
    )
    assert APP_VERSION in captured["headers"]["User-Agent"]


def test_record_submission_writes_timestamp(tmp_path: Path):
    """The frontend promises 'submitted on <date>' chips; verify the
    timestamp lands in the history entry so that UI is implementable."""
    leaderboard.record_submission(
        tmp_path,
        leaderboard.SubmissionResult(
            endpoint="x", payload={"repo_id": "u/r"}, response={},
        ),
    )
    history = leaderboard.list_submissions(tmp_path)
    assert "submitted_at" in history[0]
    # ISO 8601 with timezone offset.
    assert "T" in history[0]["submitted_at"]


def test_record_submission_serialises_concurrent_writes(tmp_path: Path):
    """Lost-update regression: with the per-process lock, two
    concurrent record_submission calls must both land in history.
    Without the lock, both threads load the same baseline and
    race on rename — last writer wins, first append disappears."""
    import threading

    barrier = threading.Barrier(2)

    def worker(payload):
        barrier.wait()
        leaderboard.record_submission(
            tmp_path,
            leaderboard.SubmissionResult(
                endpoint="x", payload={"repo_id": payload}, response={},
            ),
        )

    a = threading.Thread(target=worker, args=("u/a",))
    b = threading.Thread(target=worker, args=("u/b",))
    a.start()
    b.start()
    a.join()
    b.join()
    history = leaderboard.list_submissions(tmp_path)
    # Both submissions must be present (order doesn't matter; just
    # that neither got dropped).
    repos = sorted(s["payload"]["repo_id"] for s in history)
    assert repos == ["u/a", "u/b"]


def test_record_submission_appends_to_history(tmp_path: Path):
    result = leaderboard.SubmissionResult(
        endpoint="https://x/submit",
        payload={"repo_id": "user/x"},
        response={"id": "1"},
    )
    leaderboard.record_submission(tmp_path, result)
    leaderboard.record_submission(
        tmp_path,
        leaderboard.SubmissionResult(
            endpoint="https://x/submit",
            payload={"repo_id": "user/y"},
            response={"id": "2"},
        ),
    )
    history = leaderboard.list_submissions(tmp_path)
    assert len(history) == 2
    assert history[0]["payload"]["repo_id"] == "user/x"
    assert history[1]["payload"]["repo_id"] == "user/y"


def test_list_submissions_empty_when_no_file(tmp_path: Path):
    assert leaderboard.list_submissions(tmp_path) == []


def test_list_submissions_resilient_to_corrupt_file(tmp_path: Path):
    (tmp_path / leaderboard.SUBMISSION_FILE).write_text("not json")
    assert leaderboard.list_submissions(tmp_path) == []


def test_record_submission_atomic_write(tmp_path: Path):
    """Tmp-then-rename means a concurrent reader can't catch a torn
    file. Verify by checking no .tmp lingers after a successful
    record."""
    result = leaderboard.SubmissionResult(
        endpoint="x", payload={"repo_id": "x"}, response={},
    )
    leaderboard.record_submission(tmp_path, result)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_configured_endpoint_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("LLM_CHAIN_LEADERBOARD_URL", raising=False)
    assert leaderboard.configured_endpoint() is None


def test_configured_endpoint_treats_empty_string_as_unset(monkeypatch):
    monkeypatch.setenv("LLM_CHAIN_LEADERBOARD_URL", "")
    assert leaderboard.configured_endpoint() is None
