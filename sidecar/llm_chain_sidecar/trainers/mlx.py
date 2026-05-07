import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

from .base import EventType, Trainer, TrainingEvent

# mlx_lm log format: "Iter N: Train loss V, Learning Rate L"
_LINE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\de.+\-]+)")


class MlxTrainer(Trainer):
    """LoRA/QLoRA fine-tuning via mlx_lm.lora subprocess on Apple Silicon."""

    def train(self) -> Iterator[TrainingEvent]:
        yield TrainingEvent(
            type=EventType.START,
            message=f"Spawning mlx_lm.lora on {self.config.model_id}",
        )
        try:
            data_dir = self._stage_data()
        except Exception as e:
            yield TrainingEvent(type=EventType.ERROR, message=f"data staging failed: {e}")
            return
        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", self.config.model_id,
            "--train", "--data", data_dir,
            "--adapter-path", self.output_dir,
            "--iters", str(self.config.epochs * 100),
            "--batch-size", str(self.config.batch_size),
            "--learning-rate", str(self.config.learning_rate),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                m = _LINE.search(line.decode("utf-8", errors="replace"))
                if m:
                    yield TrainingEvent(
                        type=EventType.STEP,
                        step=int(m.group(1)),
                        total_steps=self.config.epochs * 100,
                        loss=float(m.group(2)),
                        lr=float(m.group(3)),
                    )
            rc = proc.wait()
            if rc != 0:
                yield TrainingEvent(type=EventType.ERROR, message=f"mlx_lm exited {rc}")
                return
        except Exception as e:
            yield TrainingEvent(type=EventType.ERROR, message=str(e))
            return
        yield TrainingEvent(
            type=EventType.DONE,
            message=f"Adapter saved to {self.output_dir}",
        )

    def _stage_data(self) -> str:
        """mlx_lm.lora's --data flag wants a directory containing train.jsonl
        and valid.jsonl. Read the user's single JSONL, split 90/10 (with at
        least one row per split), and write it under output_dir/_mlx_data.
        """
        src = Path(self.config.dataset_path)
        rows = [ln for ln in src.read_text().splitlines() if ln.strip()]
        if not rows:
            raise ValueError(f"No non-empty rows in {src}")
        if len(rows) == 1:
            train, valid = rows, rows
        else:
            cut = max(1, int(len(rows) * 0.9))
            cut = min(cut, len(rows) - 1)
            train, valid = rows[:cut], rows[cut:]
        staged = Path(self.output_dir) / "_mlx_data"
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "train.jsonl").write_text("\n".join(train) + "\n")
        (staged / "valid.jsonl").write_text("\n".join(valid) + "\n")
        return str(staged)
