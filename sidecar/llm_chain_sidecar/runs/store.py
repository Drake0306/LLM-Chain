from pathlib import Path

from .types import Run, RunConfig, RunStatus


class RunStore:
    """JSON-on-disk store at <root>/<run_id>/run.json."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, config: RunConfig) -> Run:
        run = Run(config=config)
        run.output_dir = str(self.root / run.id)
        Path(run.output_dir).mkdir(parents=True, exist_ok=True)
        self._write(run)
        return run

    def get(self, run_id: str) -> Run:
        return Run.model_validate_json((self.root / run_id / "run.json").read_text())

    def list(self) -> list[Run]:
        runs = [self.get(p.name) for p in self.root.iterdir() if (p / "run.json").exists()]
        return sorted(runs, key=lambda r: (r.created_at, r.created_seq), reverse=True)

    def update_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        run = self.get(run_id)
        run.status = status
        if error:
            run.error = error
        self._write(run)

    def _write(self, run: Run) -> None:
        (Path(run.output_dir) / "run.json").write_text(run.model_dump_json(indent=2))
