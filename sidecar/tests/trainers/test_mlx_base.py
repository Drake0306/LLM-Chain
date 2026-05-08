"""Contract tests for MlxSubprocessTrainer.

These exercise the pieces shared between MlxTrainer and MlxVlmTrainer —
the stdout pump, the error-tail surfacing, the GeneratorExit-safe
subprocess reaping — without depending on a real mlx_lm install. A
minimal ScriptedMlxTrainer subclass plays the role of either concrete
trainer; tests in test_mlx.py and test_mlx_vlm.py still cover the
subclass-specific bits (module name, allowed formats, row coercion).
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_chain_sidecar.datasets.types import DatasetFormat
from llm_chain_sidecar.runs.types import RunConfig
from llm_chain_sidecar.trainers._mlx_base import MlxSubprocessTrainer
from llm_chain_sidecar.trainers.base import EventType


class _ScriptedMlxTrainer(MlxSubprocessTrainer):
    """A pretend MLX trainer that accepts any text format and emits one
    row per loader entry as ``{"text": ...}``. Lets the base-class tests
    use the same simple JSONL fixtures regardless of which concrete
    trainer they're standing in for."""

    _module_name = "fake_mlx"
    _staged_dir_name = "_fake_data"

    @property
    def _done_message(self) -> str:
        return f"Done at {self.output_dir}"

    def _validate_format(self, ds_format: DatasetFormat) -> None:
        if ds_format == DatasetFormat.JSONL_CHAT_VISION:
            raise ValueError("vision format not supported in this test trainer")

    def _row_to_mlx(self, row: dict, ds_format: DatasetFormat) -> dict:
        if ds_format == DatasetFormat.JSONL_CHAT:
            return {"messages": row["messages"]}
        return {"text": str(row.get("text", ""))}


def _write_chat_jsonl(path: Path, n: int = 3) -> Path:
    lines = [
        json.dumps({"messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]})
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is darwin-only by registration")
def test_mlx_base_invokes_subcommand_form_with_module_name(tmp_path):
    """The deprecated ``python -m mlx_lm.lora`` invocation noisily warned
    on every run. Subclasses configure the supported subcommand form by
    setting ``_module_name``; the base routes that into the cmd."""
    data = _write_chat_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="lora",
                    dataset_path=str(data), epochs=1)
    trainer = _ScriptedMlxTrainer(cfg, output_dir=str(out))

    captured: list[str] = []
    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = iter([b""])
    fake_proc.wait.return_value = 0

    def fake_popen(cmd, **_kw):
        captured.extend(cmd)
        return fake_proc

    with patch(
        "llm_chain_sidecar.trainers._mlx_base.subprocess.Popen",
        side_effect=fake_popen,
    ):
        list(trainer.train())

    # `python -m fake_mlx lora` shape, not `python -m fake_mlx.lora`.
    assert "fake_mlx" in captured
    idx = captured.index("fake_mlx")
    assert captured[idx + 1] == "lora"
    assert "fake_mlx.lora" not in captured


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is darwin-only by registration")
def test_mlx_base_subprocess_reaped_on_generator_close(tmp_path):
    """Closing the generator (SSE disconnect) used to leak the OS process
    until the user manually killed it. The base's finally block now
    terminates → grace wait → kill on every exit path."""
    data = _write_chat_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="lora",
                    dataset_path=str(data), epochs=1)
    trainer = _ScriptedMlxTrainer(cfg, output_dir=str(out))

    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    # Long-running stream: many lines so the consumer has events to read
    # before abandoning.
    fake_proc.stdout.readline.side_effect = iter([b"Loading\n"] * 5 + [b""])
    fake_proc.poll.side_effect = [None, 0]  # alive, then dead after terminate
    fake_proc.wait.return_value = 0

    with patch(
        "llm_chain_sidecar.trainers._mlx_base.subprocess.Popen",
        return_value=fake_proc,
    ):
        gen = trainer.train()
        next(gen)  # START — pre-Popen
        next(gen)  # advance into the read loop so the try/finally is active
        gen.close()

    fake_proc.terminate.assert_called_once()


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is darwin-only by registration")
def test_mlx_base_surfaces_subprocess_tail_on_nonzero_exit(tmp_path):
    """When the subprocess fails, the user needs the actual error in the
    UI — not just 'exited 1'. The base buffers the last 60 lines and
    folds them into the ERROR message."""
    data = _write_chat_jsonl(tmp_path / "data.jsonl")
    out = tmp_path / "out"
    out.mkdir()
    cfg = RunConfig(model_id="m", backend="mlx", technique="lora",
                    dataset_path=str(data), epochs=1)
    trainer = _ScriptedMlxTrainer(cfg, output_dir=str(out))

    fake_proc = MagicMock()
    fake_proc.stdout = MagicMock()
    fake_proc.stdout.readline.side_effect = iter([
        b"Loading pretrained model\n",
        b"Traceback (most recent call last):\n",
        b"ValueError: something specific\n",
        b"",
    ])
    fake_proc.wait.return_value = 1
    with patch(
        "llm_chain_sidecar.trainers._mlx_base.subprocess.Popen",
        return_value=fake_proc,
    ):
        events = list(trainer.train())

    err = next(e for e in events if e.type == EventType.ERROR)
    assert "fake_mlx exited 1" in (err.message or "")
    assert "ValueError: something specific" in (err.message or "")


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX is darwin-only by registration")
def test_mlx_base_validate_format_blocks_unsupported_format(tmp_path):
    """Subclasses set the policy via _validate_format. The scripted
    trainer rejects vision; the base must propagate that as an
    ERROR event from the staging step."""
    out = tmp_path / "out"
    out.mkdir()
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})
        + "\n"
    )
    cfg = RunConfig(model_id="m", backend="mlx", technique="lora",
                    dataset_path=str(data), dataset_format="jsonl_chat_vision",
                    epochs=1)
    trainer = _ScriptedMlxTrainer(cfg, output_dir=str(out))

    events = list(trainer.train())
    assert events[0].type == EventType.START
    assert events[-1].type == EventType.ERROR
    assert "vision format not supported" in (events[-1].message or "")
