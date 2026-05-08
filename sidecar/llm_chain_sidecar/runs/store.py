from pathlib import Path
from threading import Lock

from .types import Run, RunConfig, RunStatus


class RunStore:
    """JSON-on-disk store at <root>/<run_id>/run.json.

    The disk is the source of truth, but ``list()`` used to read every
    run's JSON on every call — and the Runs page hits that endpoint on
    every navigation, so a user with hundreds of runs noticeably stalled
    the page. We keep an in-memory list cache, invalidated whenever a
    run is created, deleted, or has its status updated. Single-process,
    single-writer (the FastAPI worker), so a simple lock + dict suffice.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache_lock = Lock()
        self._list_cache: list[Run] | None = None

    def create(self, config: RunConfig) -> Run:
        run = Run(config=config)
        run.output_dir = str(self.root / run.id)
        Path(run.output_dir).mkdir(parents=True, exist_ok=True)
        self._write(run)
        self._invalidate()
        return run

    def get(self, run_id: str) -> Run:
        return Run.model_validate_json((self.root / run_id / "run.json").read_text())

    def list(self) -> list[Run]:
        with self._cache_lock:
            if self._list_cache is not None:
                return self._list_cache
            runs = [
                self.get(p.name)
                for p in self.root.iterdir()
                if (p / "run.json").exists()
            ]
            runs.sort(key=lambda r: (r.created_at, r.created_seq), reverse=True)
            self._list_cache = runs
            return runs

    def update_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        run = self.get(run_id)
        run.status = status
        if error:
            run.error = error
        self._write(run)
        self._invalidate()

    def delete(self, run_id: str) -> None:
        """Remove the run dir from disk. Caller is responsible for
        verifying the run isn't in flight; the store has no view of
        the executor's in-memory state."""
        import shutil

        run_dir = self.root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"run {run_id} not found at {run_dir}")
        shutil.rmtree(run_dir)
        self._invalidate()

    def _write(self, run: Run) -> None:
        (Path(run.output_dir) / "run.json").write_text(run.model_dump_json(indent=2))

    def _invalidate(self) -> None:
        with self._cache_lock:
            self._list_cache = None
