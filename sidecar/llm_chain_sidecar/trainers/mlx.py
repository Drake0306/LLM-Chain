import re
import subprocess
import sys
from collections.abc import Iterator

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
        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", self.config.model_id,
            "--train", "--data", self.config.dataset_path,
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
