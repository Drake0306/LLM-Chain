import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_chain_sidecar.runs.scheduler import Scheduler
from llm_chain_sidecar.runs.store import RunStore
from llm_chain_sidecar.runs.types import RunConfig


def _cfg() -> RunConfig:
    return RunConfig(
        model_id="m",
        backend="cpu",
        technique="lora",
        dataset_path="/tmp/x",
        dataset_format="jsonl_chat",
        epochs=1,
    )


def test_schedule_persists_entry_to_disk(tmp_path: Path):
    runs_root = tmp_path / "runs"
    sched_dir = tmp_path / "scheduled"
    sched = Scheduler(store=RunStore(root=runs_root), scheduled_dir=sched_dir)
    try:
        start_at = datetime.now(timezone.utc) + timedelta(seconds=60)
        entry = sched.schedule(_cfg(), start_at)
        files = list(sched_dir.iterdir())
        assert len(files) == 1
        body = json.loads(files[0].read_text())
        assert body["id"] == entry["id"]
        assert body["start_at"] == entry["start_at"]
    finally:
        sched.shutdown()


def test_schedule_rejects_past_start_at_without_fire_flag(tmp_path: Path):
    sched = Scheduler(
        store=RunStore(root=tmp_path / "runs"),
        scheduled_dir=tmp_path / "scheduled",
    )
    try:
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        with pytest.raises(ValueError, match="in the past"):
            sched.schedule(_cfg(), past)
    finally:
        sched.shutdown()


def test_schedule_accepts_past_start_at_with_fire_if_missed(tmp_path: Path):
    sched = Scheduler(
        store=RunStore(root=tmp_path / "runs"),
        scheduled_dir=tmp_path / "scheduled",
    )
    try:
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        entry = sched.schedule(_cfg(), past, fire_if_missed=True)
        assert entry["fire_if_missed"] is True
    finally:
        sched.shutdown()


def test_cancel_removes_disk_entry_and_returns_true(tmp_path: Path):
    sched = Scheduler(
        store=RunStore(root=tmp_path / "runs"),
        scheduled_dir=tmp_path / "scheduled",
    )
    try:
        entry = sched.schedule(
            _cfg(), datetime.now(timezone.utc) + timedelta(seconds=120)
        )
        assert sched.cancel(entry["id"]) is True
        assert sched.cancel(entry["id"]) is False  # idempotent
        assert list((tmp_path / "scheduled").iterdir()) == []
    finally:
        sched.shutdown()


def test_list_scheduled_orders_by_start_at(tmp_path: Path):
    sched = Scheduler(
        store=RunStore(root=tmp_path / "runs"),
        scheduled_dir=tmp_path / "scheduled",
    )
    try:
        now = datetime.now(timezone.utc)
        sched.schedule(_cfg(), now + timedelta(seconds=300))
        sched.schedule(_cfg(), now + timedelta(seconds=120))
        entries = sched.list_scheduled()
        assert len(entries) == 2
        assert entries[0]["start_at"] < entries[1]["start_at"]
    finally:
        sched.shutdown()


def test_load_persisted_recreates_timers(tmp_path: Path):
    """Restart simulation: persist an entry with one Scheduler, then
    load_persisted() with a fresh Scheduler over the same dir and
    verify it's now armed.
    """
    sched_dir = tmp_path / "scheduled"
    runs_root = tmp_path / "runs"

    first = Scheduler(store=RunStore(root=runs_root), scheduled_dir=sched_dir)
    first.schedule(_cfg(), datetime.now(timezone.utc) + timedelta(seconds=600))
    first.shutdown()

    second = Scheduler(store=RunStore(root=runs_root), scheduled_dir=sched_dir)
    try:
        loaded = second.load_persisted()
        assert loaded == 1
        assert len(second._timers) == 1
    finally:
        second.shutdown()


def test_timer_fires_and_creates_run(tmp_path: Path, monkeypatch):
    """End-to-end: a near-future schedule should fire its timer,
    create a run via the store, and invoke the executor drain.
    """
    runs_root = tmp_path / "runs"
    store = RunStore(root=runs_root)
    drained: list[str] = []
    fired = threading.Event()

    def _drain(run_id: str) -> None:
        drained.append(run_id)
        fired.set()

    sched = Scheduler(
        store=store,
        scheduled_dir=tmp_path / "scheduled",
        executor_drain=_drain,
    )
    try:
        sched.schedule(
            _cfg(),
            datetime.now(timezone.utc) + timedelta(milliseconds=100),
            fire_if_missed=True,
        )
        # The drain runs on a background thread; give the timer a
        # generous 5 s window so a slow CI runner doesn't flake.
        assert fired.wait(timeout=5.0)
        assert len(drained) == 1
        # The drain target must be a real run id the store created.
        runs = store.list()
        assert len(runs) == 1
        assert runs[0].id == drained[0]
    finally:
        sched.shutdown()


def test_schedule_rejects_naive_datetime(tmp_path: Path):
    """Naive (no-tz) datetimes are ambiguous on the wire — astimezone
    would silently coerce to local time, scheduling for a moment the
    user didn't actually pick. Reject upfront so the route surfaces
    a 400 instead of letting the schedule slip."""
    sched = Scheduler(
        store=RunStore(root=tmp_path / "runs"),
        scheduled_dir=tmp_path / "scheduled",
    )
    try:
        naive = datetime(2099, 1, 1, 12, 0, 0)  # no tzinfo
        with pytest.raises(ValueError, match="timezone"):
            sched.schedule(_cfg(), naive)
    finally:
        sched.shutdown()


def test_list_scheduled_filters_error_sidecars(tmp_path: Path):
    """``<id>.error.json`` files written by ``_fire`` on failure must
    not contaminate the listing — the UI would render them with an
    "Invalid Date" badge and no way to dismiss."""
    sched_dir = tmp_path / "scheduled"
    sched_dir.mkdir(parents=True)
    # One real entry…
    (sched_dir / "abc.json").write_text(
        json.dumps({
            "id": "abc",
            "start_at": "2099-01-01T00:00:00+00:00",
            "fire_if_missed": False,
            "config": _cfg().model_dump(mode="json"),
            "created_at": "2099-01-01T00:00:00+00:00",
        })
    )
    # …and one error sidecar that should be filtered.
    (sched_dir / "xyz.error.json").write_text(
        json.dumps({"id": "xyz", "error": "boom"})
    )

    sched = Scheduler(
        store=RunStore(root=tmp_path / "runs"),
        scheduled_dir=sched_dir,
    )
    try:
        listed = sched.list_scheduled()
        assert len(listed) == 1
        assert listed[0]["id"] == "abc"
        # ``list_errors`` surfaces the sidecar separately.
        errors = sched.list_errors()
        assert len(errors) == 1
        assert errors[0]["error"] == "boom"
    finally:
        sched.shutdown()


def test_cancel_during_fire_does_not_create_run(tmp_path: Path):
    """TOCTOU regression: a cancel that lands after _fire's lock-pop
    but before its run-create must not result in a created run.
    The fix renames the entry to .firing.json under the lock so a
    subsequent cancel can no longer find it; conversely, a cancel
    that runs first removes the file so _fire's rename misses.
    """
    runs_root = tmp_path / "runs"
    store = RunStore(root=runs_root)
    sched_dir = tmp_path / "scheduled"
    sched = Scheduler(
        store=store,
        scheduled_dir=sched_dir,
        executor_drain=lambda rid: None,
    )
    try:
        entry = sched.schedule(
            _cfg(), datetime.now(timezone.utc) + timedelta(seconds=120)
        )
        # Cancel before timer fires — _fire's rename should fail and
        # the run should not be created.
        assert sched.cancel(entry["id"]) is True
        # Simulate the timer firing late (manually invoke).
        sched._fire(entry["id"])
        assert store.list() == []
    finally:
        sched.shutdown()


def test_load_persisted_skips_missed_entries_without_fire_flag(tmp_path: Path):
    """Past-due entries without fire_if_missed should be left parked,
    not silently re-armed and fired immediately on startup."""
    sched_dir = tmp_path / "scheduled"
    sched_dir.mkdir(parents=True)
    past_entry = {
        "id": "abc",
        "start_at": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
        "fire_if_missed": False,
        "config": _cfg().model_dump(mode="json"),
        "created_at": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
    }
    (sched_dir / "abc.json").write_text(json.dumps(past_entry))

    sched = Scheduler(
        store=RunStore(root=tmp_path / "runs"),
        scheduled_dir=sched_dir,
    )
    try:
        loaded = sched.load_persisted()
        assert loaded == 0  # parked, not armed
        # Listing still surfaces the entry — the UI shows it as missed.
        assert len(sched.list_scheduled()) == 1
    finally:
        sched.shutdown()
