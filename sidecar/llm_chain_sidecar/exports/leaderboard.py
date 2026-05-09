"""Public adapter leaderboard submission (F-C14, phase 1).

Opt-in publish flow: after a user pushes a run to HF Hub, they can
submit the public repo URL + base model + eval results to a small
community leaderboard. This module owns the wire format and the
HTTP POST; the leaderboard *server* is intentionally not part of
this codebase — the endpoint URL comes from
``LLM_CHAIN_LEADERBOARD_URL`` so the user (or a deployment) can
point at whichever JSON sink they want, including a static-site
ingestion endpoint for a future read-only leaderboard frontend.

Phase 1 deliberately stops at "submit and record". Phase 2 (the
leaderboard frontend) is documented as future work in the HANDOFF
because hosting + moderation costs aren't covered by this scope.

Privacy stance: the payload is minimal — repo id, base model id,
optional eval result summary, optional notes, plus an app-version
header. We never include the user's local dataset path or any
piece of their notes that they didn't explicitly hand to this
endpoint via the route body. The route layer enforces the same.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Persisted alongside run state so the UI can show "submitted on
# <date> to <endpoint>" without re-asking the user.
SUBMISSION_FILE = "leaderboard.json"


class LeaderboardNotConfiguredError(RuntimeError):
    """Raised when the env var that points at the leaderboard endpoint
    isn't set. Surfaces as a 400 with a clear hint instead of a generic
    network error."""


class LeaderboardSubmissionError(RuntimeError):
    """Raised when the leaderboard endpoint returned non-2xx or the
    request failed. Carries the endpoint's error body so the caller
    can show it verbatim."""


@dataclass
class SubmissionResult:
    """What a successful submission looks like.

    ``response`` is the body the leaderboard returned (typically a
    moderation-pending id); we surface it back so the UI can show
    a "we'll review and publish" status without inventing one.
    """

    endpoint: str
    payload: dict
    response: dict


def configured_endpoint() -> str | None:
    """Read the leaderboard URL from env. ``None`` means submission
    is disabled — surfacing this in the UI lets the user point at
    a different endpoint without rebuilding the app."""
    return os.environ.get("LLM_CHAIN_LEADERBOARD_URL") or None


def build_payload(
    *,
    run_id: str,
    repo_id: str,
    base_model: str,
    eval_results: dict | None = None,
    notes: str | None = None,
    license_name: str | None = None,
    app_version: str = "0.1.0",
) -> dict:
    """Assemble the JSON payload sent to the leaderboard.

    Validates the inputs that we care about — empty strings get
    promoted to None so the receiving server doesn't store
    placeholders. ``eval_results`` is opaque from this module's
    perspective; the leaderboard server decides what shape to
    accept.
    """
    payload: dict = {
        "run_id": run_id,
        "repo_id": repo_id.strip(),
        "base_model": base_model.strip(),
        "app_version": app_version,
    }
    if not payload["repo_id"]:
        raise ValueError("repo_id is required for leaderboard submission.")
    if not payload["base_model"]:
        raise ValueError("base_model is required for leaderboard submission.")
    if eval_results:
        payload["eval_results"] = eval_results
    if notes and notes.strip():
        payload["notes"] = notes.strip()
    if license_name and license_name.strip():
        payload["license"] = license_name.strip()
    return payload


def submit(
    payload: dict,
    *,
    endpoint: str | None = None,
    requester=None,
    timeout: float = 10.0,
) -> SubmissionResult:
    """POST ``payload`` to the configured leaderboard.

    ``requester`` lets tests inject a fake function to capture the
    call without hitting the network. Default is ``requests.post``.

    Raises:
    - LeaderboardNotConfiguredError: env var unset and no override.
    - LeaderboardSubmissionError: non-2xx response or transport error.
    """
    target = endpoint or configured_endpoint()
    if not target:
        raise LeaderboardNotConfiguredError(
            "Leaderboard endpoint isn't configured. Set "
            "LLM_CHAIN_LEADERBOARD_URL in the sidecar's environment "
            "to enable submissions."
        )

    if requester is None:
        import requests as _requests

        requester = _requests.post

    try:
        resp = requester(
            target,
            json=payload,
            timeout=timeout,
            headers={"User-Agent": "LLM-Chain/0.1.0 (leaderboard)"},
        )
    except Exception as e:  # noqa: BLE001 — network errors map to one shape
        raise LeaderboardSubmissionError(
            f"Leaderboard POST failed: {e}"
        ) from e

    status = getattr(resp, "status_code", None)
    if status is None or status < 200 or status >= 300:
        body = ""
        try:
            body = resp.text  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
        raise LeaderboardSubmissionError(
            f"Leaderboard returned {status}: {body[:300]}"
        )

    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 — leaderboard might return non-JSON 200
        body = {}

    return SubmissionResult(endpoint=target, payload=payload, response=body)


def record_submission(run_dir: Path, result: SubmissionResult) -> None:
    """Persist the submission to ``<run_dir>/leaderboard.json``.

    Append-style: the same run can be re-submitted (after improving
    the model, switching to a public repo) and we keep every entry
    so the UI can show history.
    """
    p = run_dir / SUBMISSION_FILE
    history: list[dict] = []
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                history = list(data.get("submissions", []))
        except json.JSONDecodeError:
            history = []
    history.append({
        "endpoint": result.endpoint,
        "payload": result.payload,
        "response": result.response,
    })
    # Atomic write so a concurrent reader can't catch a torn file.
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"submissions": history}, indent=2))
    os.replace(tmp, p)


def list_submissions(run_dir: Path) -> list[dict]:
    p = run_dir / SUBMISSION_FILE
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    return list(data.get("submissions", []))


__all__ = [
    "LeaderboardNotConfiguredError",
    "LeaderboardSubmissionError",
    "SUBMISSION_FILE",
    "SubmissionResult",
    "build_payload",
    "configured_endpoint",
    "list_submissions",
    "record_submission",
    "submit",
]
