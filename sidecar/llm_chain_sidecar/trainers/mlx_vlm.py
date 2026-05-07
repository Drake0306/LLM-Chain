"""LoRA fine-tuning for vision-language models on Apple Silicon via mlx-vlm.

Mirrors the text-MLX trainer: subprocess `mlx_vlm.lora` and parse its stdout
for loss/step events. mlx-vlm uses the same iter-based log line format as
mlx_lm so the regex carries over.
"""
import re
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path

from .base import EventType, Trainer, TrainingEvent

# mlx_vlm log format mirrors mlx_lm: "Iter N: Train loss V, Learning Rate L".
_LINE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\de.+\-]+)")

_TAIL_LINES = 60


class MlxVlmTrainer(Trainer):
    def train(self) -> Iterator[TrainingEvent]:
        yield TrainingEvent(
            type=EventType.START,
            message=f"Spawning mlx_vlm.lora on {self.config.model_id}",
        )
        try:
            data_dir = self._stage_data()
        except Exception as e:
            yield TrainingEvent(type=EventType.ERROR, message=f"data staging failed: {e}")
            return
        cmd = [
            sys.executable, "-m", "mlx_vlm.lora",
            "--model", self.config.model_id,
            "--train", "--data", data_dir,
            "--adapter-path", self.output_dir,
            "--iters", str(self.config.epochs * 100),
            "--batch-size", str(self.config.batch_size),
            "--learning-rate", str(self.config.learning_rate),
            # mlx_vlm inherits the --num-layers 16 default from mlx_lm; -1 = all
            # layers. See trainers/mlx.py for the why.
            "--num-layers", "-1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Watcher thread terminates the subprocess when cancellation fires —
        # readline() blocks on the main thread otherwise.
        def _watch_cancel():
            self.cancel_event.wait()
            if proc.poll() is None:
                proc.terminate()

        threading.Thread(target=_watch_cancel, daemon=True).start()

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
                    message=f"mlx_vlm exited {rc}:\n{detail}",
                )
                return
        except Exception as e:
            yield TrainingEvent(type=EventType.ERROR, message=str(e))
            return
        yield TrainingEvent(
            type=EventType.DONE,
            message=f"VLM adapter saved to {self.output_dir}",
        )

    def _stage_data(self) -> str:
        """mlx_vlm.lora --data wants a directory with train.jsonl + valid.jsonl,
        same as mlx_lm. We re-emit the user's chat-vision JSONL into both.
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
        staged = Path(self.output_dir) / "_mlx_vlm_data"
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "train.jsonl").write_text("\n".join(train) + "\n")
        (staged / "valid.jsonl").write_text("\n".join(valid) + "\n")
        return str(staged)
