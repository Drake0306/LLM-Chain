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
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Persisted alongside run state so the UI can show "submitted on
# <date> to <endpoint>" without re-asking the user.
SUBMISSION_FILE = "leaderboard.json"

# Per-process write lock for the submission history. Atomic
# tmp-then-rename prevents torn reads but doesn't prevent
# lost-update races: two concurrent record_submission calls would
# both load the existing list, both append, both rename — the
# second one's append wipes the first. The lock serialises the
# read-modify-write for the lifetime of this process. (A second
# sidecar process touching the same file would still race; that's
# a much rarer scenario for a single-user desktop app.)
_history_lock = threading.Lock()

# Sensitive payload keys we refuse to forward to the leaderboard.
# The route layer accepts ``eval_results: dict`` opaque-style so a
# misled UI / curl could put dataset_path or env values in there.
# Strip them before the payload leaves the box.
_SENSITIVE_PAYLOAD_KEYS = frozenset(
    {
        "dataset_path",
        "output_dir",
        "user",
        "username",
        "home",
        "path",
        "secret",
        "token",
        "api_key",
        "password",
    }
)


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


def _scrub_sensitive(d: dict) -> dict:
    """Drop keys that would leak local context (paths, secrets) from
    an opaque user-supplied dict. Compares case-insensitively
    against a known-bad list. The leaderboard server decides what
    shape to accept; this is a defence-in-depth filter so a UI bug
    or curl typo doesn't ship the user's ``~/Documents`` paths to
    a third-party endpoint.
    """
    out: dict = {}
    for k, v in d.items():
        if not isinstance(k, str):
            continue
        if k.lower() in _SENSITIVE_PAYLOAD_KEYS:
            continue
        # Recurse into nested dicts so {"meta": {"dataset_path": ...}}
        # is also scrubbed.
        if isinstance(v, dict):
            out[k] = _scrub_sensitive(v)
        else:
            out[k] = v
    return out


def build_payload(
    *,
    run_id: str,
    repo_id: str,
    base_model: str,
    eval_results: dict | None = None,
    notes: str | None = None,
    license_name: str | None = None,
    app_version: str | None = None,
) -> dict:
    """Assemble the JSON payload sent to the leaderboard.

    Validates the inputs that we care about — empty strings get
    promoted to None so the receiving server doesn't store
    placeholders. ``eval_results`` runs through ``_scrub_sensitive``
    so a UI bug can't smuggle local paths past the privacy
    boundary. ``app_version`` defaults to the live recipes.APP_VERSION
    so the payload's version field tracks releases instead of the
    stale literal it used to ship.
    """
    if app_version is None:
        # Late import to dodge a circular dep — recipes imports this
        # module's exports symbol via the package re-export.
        from llm_chain_sidecar.recipes import APP_VERSION as _APP_VERSION

        app_version = _APP_VERSION

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
        scrubbed = _scrub_sensitive(eval_results)
        if scrubbed:
            payload["eval_results"] = scrubbed
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

    # Pull the live version so the User-Agent doesn't lie when v0.2
    # ships. Late import to dodge a circular dep through the package
    # re-export.
    from llm_chain_sidecar.recipes import APP_VERSION as _APP_VERSION

    try:
        resp = requester(
            target,
            json=payload,
            timeout=timeout,
            headers={
                "User-Agent": f"LLM-Chain/{_APP_VERSION} (leaderboard)",
            },
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
    so the UI can show history. Each entry carries a UTC ISO
    ``submitted_at`` timestamp so the UI can render "submitted on
    <date>" without re-asking the server.

    Concurrency: the read-modify-write is wrapped in a per-process
    lock so two near-simultaneous submissions don't both load the
    history, both append, and have one entry silently disappear at
    rename time. (Atomic tmp-then-rename only protects readers from
    seeing a half-written file; the lost-update race is orthogonal.)
    """
    p = run_dir / SUBMISSION_FILE
    entry = {
        "endpoint": result.endpoint,
        "payload": result.payload,
        "response": result.response,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    with _history_lock:
        history: list[dict] = []
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, dict):
                    history = list(data.get("submissions", []))
            except json.JSONDecodeError:
                history = []
        history.append(entry)
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
