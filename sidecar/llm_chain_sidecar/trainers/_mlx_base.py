"""Template-method base for the two MLX subprocess trainers.

MlxTrainer (``mlx_lm lora``) and MlxVlmTrainer (``mlx_vlm lora``) used to
inline the same subprocess lifecycle three times: spawn, watcher thread
that terminates on cancel, line-by-line stdout pump with regex parsing,
non-zero-exit error reporting with stdout tail, and the
GeneratorExit-safe ``terminate → wait → kill`` cleanup. ``MlxSubprocessTrainer``
owns all of that. Subclasses configure two things — what module to spawn
and how to coerce loader rows into the JSONL mlx_lm/mlx_vlm expects.

If you find yourself adding *behavior* to a subclass beyond those two
hooks, you're probably reaching for a feature that should live here
instead so both trainers benefit (e.g. download-progress forwarding when
mlx_lm gains it).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from abc import abstractmethod
from collections import deque
from collections.abc import Iterator
from pathlib import Path

from llm_chain_sidecar.datasets import DatasetFormat, load_dataset, make_source

from .base import EventType, Trainer, TrainingEvent

# mlx_lm and mlx_vlm share the same per-step log format:
# "Iter N: Train loss V, Learning Rate L". One regex covers both.
_LINE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\de.+\-]+)")

# How many recent stdout lines we keep so we can surface the real
# subprocess error back to the UI when it exits non-zero. Tracebacks max
# out around 30-40 lines for the failure modes seen in user logs.
_TAIL_LINES = 60


class MlxSubprocessTrainer(Trainer):
    """Drive ``python -m <module> lora ...`` as a subprocess.

    Subclasses set the class attributes below and implement two hooks:
      - ``_module_name`` (e.g. ``"mlx_lm"``)
      - ``_staged_dir_name`` (e.g. ``"_mlx_data"``)
      - ``_done_message``: text for the terminal DONE event
      - ``_validate_format(ds_format)``: raise ValueError when this trainer
        can't handle the chosen dataset format
      - ``_row_to_mlx(row, ds_format)``: produce the dict that goes into
        train.jsonl / valid.jsonl

    Everything else — process lifecycle, error tail capture, cancel
    propagation, and orphan-subprocess prevention — lives here.
    """

    # Filled in by subclasses.
    _module_name: str
    _staged_dir_name: str
    _done_message: str

    @abstractmethod
    def _validate_format(self, ds_format: DatasetFormat) -> None:
        """Raise ValueError if ``ds_format`` doesn't belong on this trainer."""

    @abstractmethod
    def _row_to_mlx(self, row: dict, ds_format: DatasetFormat) -> dict:
        """Coerce a loader row into the JSONL shape mlx expects."""

    def train(self) -> Iterator[TrainingEvent]:
        yield TrainingEvent(
            type=EventType.START,
            message=f"Spawning {self._module_name}.lora on {self.config.model_id}",
        )
        try:
            data_dir = self._stage_data()
        except Exception as e:
            yield TrainingEvent(type=EventType.ERROR, message=f"data staging failed: {e}")
            return

        cmd = self._build_cmd(data_dir)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Watcher thread terminates the subprocess when cancellation fires.
        # Done in the background so the main loop's blocking readline()
        # doesn't delay cancellation by a step's worth of stdout.
        def _watch_cancel() -> None:
            self.cancel_event.wait()
            if proc.poll() is None:
                proc.terminate()

        threading.Thread(target=_watch_cancel, daemon=True).start()

        # Last N lines, used to surface the real error if the subprocess
        # exits non-zero. Without this the user just sees "exited 1".
        tail: deque[str] = deque(maxlen=_TAIL_LINES)

        try:
            try:
                yield from self._pump_stdout(proc, tail)
                rc = proc.wait()
                if self.is_canceled():
                    yield TrainingEvent(type=EventType.CANCELED, message="Canceled by user")
                    return
                if rc != 0:
                    detail = "\n".join(tail) if tail else "(no output captured)"
                    yield TrainingEvent(
                        type=EventType.ERROR,
                        message=f"{self._module_name} exited {rc}:\n{detail}",
                    )
                    return
            except Exception as e:  # noqa: BLE001 — surfaced as a single ERROR event
                yield TrainingEvent(type=EventType.ERROR, message=str(e))
                return
            yield TrainingEvent(type=EventType.DONE, message=self._done_message)
        finally:
            self._reap(proc)

    # --- Internals ---------------------------------------------------------

    def _build_cmd(self, data_dir: str) -> list[str]:
        """Construct the ``python -m <module> lora ...`` invocation.

        Newer mlx_lm versions deprecate the ``python -m mlx_lm.lora`` form
        in favor of the ``mlx_lm`` package + ``lora`` subcommand. The old
        form still works but printed a noisy banner that ended up at the
        head of every captured error tail in the user's run logs.
        """
        # Iteration count: max_steps (when set) overrides the
        # epochs-based default. The LR finder uses this to spawn
        # short 10-iter sniff runs without monkey-patching the
        # trainer; ordinary runs leave it None and get the historic
        # ``epochs * 100`` budget.
        iters = (
            self.config.max_steps
            if self.config.max_steps is not None
            else self.config.epochs * 100
        )
        cmd = [
            sys.executable, "-m", self._module_name, "lora",
            "--model", self.config.model_id,
            "--train", "--data", data_dir,
            "--adapter-path", self.output_dir,
            "--iters", str(iters),
            "--batch-size", str(self.config.batch_size),
            "--learning-rate", str(self.config.learning_rate),
            # mlx_lm/mlx_vlm default to --num-layers 16, which raises
            # ValueError on any model with fewer than 16 transformer
            # layers (Pythia-70m has 6). -1 = "train LoRA on all layers"
            # and works for any model size.
            "--num-layers", "-1",
        ]
        # Resume support — the route layer has already validated that
        # ``resume_from`` references a SUCCEEDED run with an adapter on
        # disk. mlx_lm reads the file's tensors at the start of training
        # and uses them as the LoRA initial values instead of random
        # initialization, so the new run picks up where the parent left
        # off (with whatever fresh data and LR the user picked).
        rid = getattr(self.config, "resume_from", None)
        if rid:
            adapter = Path(self.output_dir).parent / rid / "adapters.safetensors"
            if not adapter.exists():
                # The parent run was likely deleted between create-time
                # validation and execute-time. Falling through to fresh
                # init would silently lose the user's intent (they
                # explicitly asked to continue from this run); fail
                # loudly so they can either restore the parent or
                # start a new run.
                raise FileNotFoundError(
                    f"Cannot resume from run {rid}: adapter file missing at "
                    f"{adapter}. The parent run may have been deleted; "
                    "start a fresh run or restore the parent."
                )
            cmd += ["--resume-adapter-file", str(adapter)]
        return cmd

    def _pump_stdout(
        self, proc: subprocess.Popen, tail: deque[str]
    ) -> Iterator[TrainingEvent]:
        """Translate stdout lines into STEP / LOG events.

        Per-iteration loss lines match ``_LINE``; everything else (model
        loading, dataset loading, tqdm bars) is forwarded as a LOG event so
        the UI can show *what* the subprocess is doing while we wait for
        the first iter — without this, the chart area would stay at
        "Waiting for first step…" indefinitely during downloads.
        """
        total_steps = self.config.epochs * 100
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue
            tail.append(text)
            m = _LINE.search(text)
            if m:
                yield TrainingEvent(
                    type=EventType.STEP,
                    step=int(m.group(1)),
                    total_steps=total_steps,
                    loss=float(m.group(2)),
                    lr=float(m.group(3)),
                )
            else:
                yield TrainingEvent(type=EventType.LOG, message=text)

    @staticmethod
    def _reap(proc: subprocess.Popen) -> None:
        """Make sure the subprocess is gone before we return.

        A generator can be close()'d at any yield, raising GeneratorExit
        — a BaseException, not Exception, so the surrounding except
        won't catch it. Without this finally, an SSE client disconnect
        or app close mid-run leaves mlx_lm running as an orphaned OS
        process consuming CPU/RAM forever. terminate → grace wait →
        kill guarantees reaping.
        """
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

    # --- Data staging ------------------------------------------------------

    def _stage_data(self) -> str:
        """Materialize train.jsonl / valid.jsonl / test.jsonl under output_dir.

        The historic implementation copied source bytes verbatim, which
        only worked when the source was already JSONL chat. CSV /
        text-dir / HF Hub datasets fell through and mlx_lm exploded with
        ``JSONDecodeError`` on plain-text lines. This routes through the
        dataset loader so every format ends up as a JSON object the
        subprocess can parse, and the per-format coercion lives in the
        subclass's ``_row_to_mlx``.
        """
        ds_format = DatasetFormat(self.config.dataset_format)
        self._validate_format(ds_format)
        rows = load_dataset(
            make_source(ds_format, self.config.dataset_path, self.config.text_column)
        )
        if not rows:
            raise ValueError(f"No rows found in {self.config.dataset_path}")
        # mlx_lm needs disjoint train and valid sets to give meaningful
        # eval losses. With one row the only split that satisfies the
        # "both non-empty" requirement is to put the same row in both,
        # which makes the eval curve a copy of the training curve and
        # gives the user no signal at all. Refuse with a clear ask
        # instead of silently producing a bad fine-tune. (mlx_lm's
        # downstream create_dataset chokes on empty splits, so
        # one-into-each isn't an option either.)
        if len(rows) < 2:
            raise ValueError(
                "Dataset has only 1 row. Training needs at least 2 so "
                "that the train and validation splits don't overlap. "
                "Add another row to your dataset and try again."
            )
        mlx_rows = [self._row_to_mlx(r, ds_format) for r in rows]

        # Split 90/10 with each subset getting at least one row, mirroring
        # mlx_lm's expectation that train.jsonl and valid.jsonl are both
        # populated.
        cut = max(1, int(len(mlx_rows) * 0.9))
        cut = min(cut, len(mlx_rows) - 1)
        train, valid = mlx_rows[:cut], mlx_rows[cut:]

        staged = Path(self.output_dir) / self._staged_dir_name
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "train.jsonl").write_text(
            "\n".join(json.dumps(r) for r in train) + "\n"
        )
        (staged / "valid.jsonl").write_text(
            "\n".join(json.dumps(r) for r in valid) + "\n"
        )
        # Deliberately do NOT write test.jsonl. mlx_lm's load_local_dataset
        # iterates ('train', 'valid', 'test') and calls
        # ``create_dataset(data, tokenizer, config)`` unconditionally on
        # whatever load_subset returns — including an empty list — and
        # the very first thing create_dataset does is ``sample = data[0]``,
        # which raises IndexError on []. The early ``if not path.exists():
        # return []`` short-circuit at the top of load_subset DOES bypass
        # this, but only when the file genuinely doesn't exist on disk.
        # Writing an empty test.jsonl defeats the short-circuit and crashes
        # training. If the test split is empty, omit the file entirely.
        # Defensive cleanup: remove any stale test.jsonl from a previous
        # build of the staged dir (re-runs reuse the parent run dir).
        stale_test = staged / "test.jsonl"
        if stale_test.exists():
            stale_test.unlink()
        return str(staged)
