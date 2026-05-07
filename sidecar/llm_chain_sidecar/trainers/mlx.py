import re
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path

from .base import EventType, Trainer, TrainingEvent

# mlx_lm log format: "Iter N: Train loss V, Learning Rate L"
_LINE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\de.+\-]+)")

# How many recent stdout lines we keep so we can surface the real mlx_lm error
# back to the UI when the subprocess exits non-zero. Tracebacks max out around
# 30-40 lines for the cases we've seen.
_TAIL_LINES = 60


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
            # mlx_lm defaults to --num-layers 16, which raises ValueError on
            # any model with fewer than 16 transformer layers (Pythia-70m has
            # 6, SmolLM2-135M has 30 but small Qwen3 variants are borderline).
            # -1 means "train LoRA on all layers" — works for any model size.
            "--num-layers", "-1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Watcher thread terminates the subprocess when cancellation fires.
        # Done in the background so the main loop's blocking readline() doesn't
        # delay cancellation by a step's worth of stdout.
        def _watch_cancel():
            self.cancel_event.wait()
            if proc.poll() is None:
                proc.terminate()

        threading.Thread(target=_watch_cancel, daemon=True).start()

        # Keep the last N lines so we can surface the real error if the
        # subprocess fails. Without this the user just sees "exited 1" and
        # has no way to act on it.
        tail: deque[str] = deque(maxlen=_TAIL_LINES)

        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                tail.append(text)
                m = _LINE.search(text)
                if m:
                    yield TrainingEvent(
                        type=EventType.STEP,
                        step=int(m.group(1)),
                        total_steps=self.config.epochs * 100,
                        loss=float(m.group(2)),
                        lr=float(m.group(3)),
                    )
            rc = proc.wait()
            if self.is_canceled():
                yield TrainingEvent(type=EventType.CANCELED, message="Canceled by user")
                return
            if rc != 0:
                detail = "\n".join(tail) if tail else "(no output captured)"
                yield TrainingEvent(
                    type=EventType.ERROR,
                    message=f"mlx_lm exited {rc}:\n{detail}",
                )
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
