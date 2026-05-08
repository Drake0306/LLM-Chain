"""Schedule a run to fire later.

Persists pending entries to ``<scheduled_dir>/<id>.json`` so a sidecar
restart doesn't lose the user's overnight kickoff. On module import
(via :func:`load_persisted`) the scheduler scans the directory and
recreates timers; entries whose start time has already passed either
fire immediately (when ``fire_if_missed=True``) or stay parked until
the user removes them.

Caveats — surfaced in the UI as a banner:
  - The sidecar must be running when the timer fires. The scheduler
    doesn't wake the laptop, doesn't talk to launchd / systemd, and
    doesn't survive an OS shutdown / sleep that kills the sidecar
    process. Closing the desktop app means scheduled runs do NOT fire.
  - There's no smart "wait for current run to finish" — when a timer
    fires while another run is RUNNING, both will compete for GPU/CPU.
    Document this and let the user space their schedules out.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .store import RunStore
from .types import RunConfig


_DEFAULT_DIR = Path.home() / ".llm-chain" / "scheduled"


def default_scheduled_dir() -> Path:
    """Resolve ``~/.llm-chain/scheduled`` honouring the test override env.

    Mirrors the runs-root override pattern in api.routes — tests set
    ``LLM_CHAIN_SCHEDULED_DIR`` to a tmp_path so they don't litter
    the user's home directory.
    """
    env = os.environ.get("LLM_CHAIN_SCHEDULED_DIR")
    if env:
        return Path(env)
    return _DEFAULT_DIR


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Scheduler:
    """Disk-backed dictionary of pending scheduled runs.

    Single-instance per process (the API constructs one in ``routes``
    on import). Methods are thread-safe behind ``_lock`` because the
    timer callback fires on a background thread.
    """

    def __init__(
        self,
        store: RunStore,
        scheduled_dir: Path | None = None,
        executor_drain: Callable[[str], None] | None = None,
    ) -> None:
        self._store = store
        self._dir = scheduled_dir or default_scheduled_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        # Caller injects a "kick the executor for this run id" function
        # so the scheduler doesn't depend on the executor module's
        # exact API. The api.routes wiring passes a closure that drains
        # ``_executor.execute(run_id)`` in a daemon thread so the run
        # makes it from PENDING to a terminal state.
        self._executor_drain = executor_drain
        self._lock = threading.Lock()
        self._timers: dict[str, threading.Timer] = {}

    # --- public surface ----------------------------------------------

    def schedule(
        self,
        config: RunConfig,
        start_at: datetime,
        fire_if_missed: bool = False,
    ) -> dict:
        """Persist a new scheduled entry and arm its timer.

        Returns the entry as a dict (id + start_at + config). Raises
        ValueError when the start_at is in the past and
        fire_if_missed is False — surfacing a "did you mean now?"
        prompt at the route layer is friendlier than silently
        dropping the schedule.
        """
        scheduled_id = uuid4().hex[:12]
        entry = {
            "id": scheduled_id,
            "start_at": start_at.astimezone(timezone.utc).isoformat(),
            "fire_if_missed": fire_if_missed,
            "config": config.model_dump(mode="json"),
            "created_at": _now().isoformat(),
        }
        delay_s = (start_at - _now()).total_seconds()
        if delay_s < 0 and not fire_if_missed:
            raise ValueError(
                "start_at is in the past. Pass fire_if_missed=True if you "
                "want this to run immediately."
            )

        path = self._dir / f"{scheduled_id}.json"
        path.write_text(json.dumps(entry, indent=2))

        with self._lock:
            self._arm_locked(scheduled_id, max(0.0, delay_s))
        return entry

    def list_scheduled(self) -> list[dict]:
        """Return every persisted entry in start-time order."""
        out: list[dict] = []
        for p in self._dir.iterdir():
            if p.suffix != ".json":
                continue
            try:
                out.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                # Torn write or hand-edited file — skip rather than
                # crashing the listing endpoint.
                continue
        out.sort(key=lambda e: e.get("start_at", ""))
        return out

    def cancel(self, scheduled_id: str) -> bool:
        """Drop the entry without firing. Returns True when the entry
        existed (and was canceled), False when it had already fired
        or never existed."""
        with self._lock:
            timer = self._timers.pop(scheduled_id, None)
            if timer is not None:
                timer.cancel()
        path = self._dir / f"{scheduled_id}.json"
        if not path.exists():
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True

    def load_persisted(self) -> int:
        """Recreate timers for every persisted entry on startup.

        Past-due entries with fire_if_missed=True fire immediately on
        load (delay 0); past-due entries without that flag stay parked
        and are returned in list_scheduled() so the UI can surface them
        with a "missed" badge.
        """
        loaded = 0
        for p in self._dir.iterdir():
            if p.suffix != ".json":
                continue
            try:
                entry = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            try:
                start_at = datetime.fromisoformat(entry["start_at"])
            except (KeyError, ValueError):
                continue
            delay_s = (start_at - _now()).total_seconds()
            if delay_s < 0 and not entry.get("fire_if_missed", False):
                continue
            with self._lock:
                self._arm_locked(entry["id"], max(0.0, delay_s))
            loaded += 1
        return loaded

    def shutdown(self) -> None:
        """Cancel every armed timer. Useful for tests; production sees
        this only on process exit, where threading.Timer threads are
        daemons that die with the process anyway."""
        with self._lock:
            for t in self._timers.values():
                t.cancel()
            self._timers.clear()

    # --- internals ---------------------------------------------------

    def _arm_locked(self, scheduled_id: str, delay_s: float) -> None:
        """Replace any existing timer for this id with a fresh one.

        Caller must hold ``self._lock``.
        """
        existing = self._timers.pop(scheduled_id, None)
        if existing is not None:
            existing.cancel()
        timer = threading.Timer(delay_s, self._fire, args=(scheduled_id,))
        timer.daemon = True
        timer.start()
        self._timers[scheduled_id] = timer

    def _fire(self, scheduled_id: str) -> None:
        """Timer callback: read the entry, create the run, kick the
        executor, then delete the schedule file.

        Errors are swallowed and surfaced through a sibling
        ``<id>.error.json`` file so the UI can render "scheduled run
        failed to start" — propagating them out of the timer thread
        would just kill the daemon without telling anyone.
        """
        path = self._dir / f"{scheduled_id}.json"
        if not path.exists():
            return
        try:
            entry = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return
        try:
            cfg = RunConfig.model_validate(entry["config"])
            run = self._store.create(cfg)
            with self._lock:
                self._timers.pop(scheduled_id, None)
            # Best-effort cleanup so the listing doesn't show fired
            # entries forever.
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            if self._executor_drain is not None:
                threading.Thread(
                    target=self._executor_drain,
                    args=(run.id,),
                    daemon=True,
                ).start()
        except Exception as e:  # noqa: BLE001 — surfaced via sidecar log
            err_path = self._dir / f"{scheduled_id}.error.json"
            try:
                err_path.write_text(
                    json.dumps(
                        {"id": scheduled_id, "error": str(e)},
                        indent=2,
                    )
                )
            except OSError:
                pass


def reset_for_tests(scheduler: Scheduler) -> None:
    """Cancel every timer and wipe the on-disk dir. Tests use this in
    a fixture to avoid cross-test scheduler state — tests share the
    module-level ``_scheduler`` via the api.routes singleton.
    """
    scheduler.shutdown()
    if scheduler._dir.exists():
        shutil.rmtree(scheduler._dir)
    scheduler._dir.mkdir(parents=True, exist_ok=True)
