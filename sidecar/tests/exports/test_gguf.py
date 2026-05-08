import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_chain_sidecar.exports import gguf as gguf_mod
from llm_chain_sidecar.exports.gguf import SUPPORTED_QUANTS, convert_to_gguf


def _stub_convert_script(tmp_path: Path, monkeypatch) -> Path:
    """Make convert_hf_to_gguf.py resolve to an existing file under tmp_path."""
    llama_dir = tmp_path / "llama.cpp"
    llama_dir.mkdir()
    (llama_dir / "convert_hf_to_gguf.py").write_text("# stub")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(llama_dir))
    return llama_dir


def _stub_quantize_binary(llama_dir: Path) -> Path:
    bin_dir = llama_dir / "build" / "bin"
    bin_dir.mkdir(parents=True)
    q = bin_dir / "llama-quantize"
    q.write_text("# stub")
    q.chmod(0o755)
    return q


def test_supported_quants_covers_plan_defaults():
    for q in ("q4_k_m", "q8_0", "f16"):
        assert q in SUPPORTED_QUANTS


def test_convert_to_gguf_rejects_unknown_quant(tmp_path: Path, monkeypatch):
    _stub_convert_script(tmp_path, monkeypatch)
    merged = tmp_path / "merged"
    merged.mkdir()
    with pytest.raises(ValueError, match="Unsupported quant"):
        convert_to_gguf(merged, quant="q42_super")


def _fake_popen_factory(calls: list[list[str]], output_writer):
    """Return a Popen-shaped fake whose readline drains immediately and whose
    wait() succeeds. Records the cmd list and lets the test materialize the
    expected output file via ``output_writer(cmd)``."""
    class FakeProc:
        def __init__(self, cmd):
            self._cmd = cmd
            self.stdout = self
            self.returncode = 0

        def readline(self):
            return b""

        def wait(self):
            output_writer(self._cmd)
            return 0

    def factory(cmd, **_kw):
        calls.append(cmd)
        return FakeProc(cmd)

    return factory


def test_convert_to_gguf_direct_outtype_runs_one_subprocess(tmp_path: Path, monkeypatch):
    _stub_convert_script(tmp_path, monkeypatch)
    merged = tmp_path / "merged"
    merged.mkdir()

    calls: list[list[str]] = []

    def write_out(cmd):
        out_idx = cmd.index("--outfile") + 1
        Path(cmd[out_idx]).write_bytes(b"\x00")

    monkeypatch.setattr(
        gguf_mod.subprocess, "Popen", _fake_popen_factory(calls, write_out)
    )
    out = convert_to_gguf(merged, quant="q8_0")

    assert out == merged.parent / "merged-q8_0.gguf"
    assert out.exists()
    assert len(calls) == 1
    cmd = calls[0]
    assert sys.executable in cmd[0]
    assert "--outtype" in cmd
    assert cmd[cmd.index("--outtype") + 1] == "q8_0"


def test_convert_to_gguf_kquant_chains_convert_then_quantize(tmp_path: Path, monkeypatch):
    llama_dir = _stub_convert_script(tmp_path, monkeypatch)
    quantize_bin = _stub_quantize_binary(llama_dir)
    merged = tmp_path / "merged"
    merged.mkdir()

    calls: list[list[str]] = []

    def write_out(cmd):
        if "--outfile" in cmd:
            out = Path(cmd[cmd.index("--outfile") + 1])
        else:
            out = Path(cmd[2])
        out.write_bytes(b"\x00")

    monkeypatch.setattr(
        gguf_mod.subprocess, "Popen", _fake_popen_factory(calls, write_out)
    )
    out = convert_to_gguf(merged, quant="q4_k_m")

    assert out == merged.parent / "merged-q4_k_m.gguf"
    assert out.exists()
    # Two subprocess calls: HF→f16 GGUF, then llama-quantize → q4_k_m.
    assert len(calls) == 2
    assert "--outtype" in calls[0] and calls[0][calls[0].index("--outtype") + 1] == "f16"
    assert calls[1][0] == str(quantize_bin)
    assert calls[1][-1] == "q4_k_m"


def test_convert_to_gguf_skips_f16_step_if_already_present(tmp_path: Path, monkeypatch):
    llama_dir = _stub_convert_script(tmp_path, monkeypatch)
    _stub_quantize_binary(llama_dir)
    merged = tmp_path / "merged"
    merged.mkdir()
    # Pre-existing f16 from a prior run should be reused (saves a long
    # convert step on a second-quant attempt).
    (merged.parent / "merged-f16.gguf").write_bytes(b"\x00")

    calls: list[list[str]] = []

    def write_out(cmd):
        if "--outfile" not in cmd:
            Path(cmd[2]).write_bytes(b"\x00")

    monkeypatch.setattr(
        gguf_mod.subprocess, "Popen", _fake_popen_factory(calls, write_out)
    )
    convert_to_gguf(merged, quant="q4_k_m")
    # Only the quantize call ran; convert was skipped.
    assert len(calls) == 1
    assert calls[0][-1] == "q4_k_m"


def test_convert_to_gguf_errors_when_bootstrap_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path / "missing"))
    merged = tmp_path / "merged"
    merged.mkdir()
    with pytest.raises(FileNotFoundError, match="convert_hf_to_gguf.py"):
        convert_to_gguf(merged, quant="f16")


def test_convert_to_gguf_forwards_progress_lines(tmp_path: Path, monkeypatch):
    """The UI shows the last subprocess line so users can see downloads,
    fusing, etc. Verify the on_progress callback receives every stdout line."""
    _stub_convert_script(tmp_path, monkeypatch)
    merged = tmp_path / "merged"
    merged.mkdir()

    # Simulate a real subprocess that emits progress lines.
    class FakeProc:
        def __init__(self, cmd, lines):
            self._cmd = cmd
            self._lines = iter(lines + [b""])
            self.stdout = self
            self.returncode = 0

        def readline(self):
            return next(self._lines, b"")

        def wait(self):
            # Materialize the file at whatever --outfile path the trainer
            # passed. The atomic-write wrapper hands subprocesses a .partial
            # path that gets renamed on success — hardcoding the final name
            # here would race the rename and trip FileNotFoundError.
            out_idx = self._cmd.index("--outfile") + 1
            Path(self._cmd[out_idx]).write_bytes(b"\x00")
            return 0

    fake_lines = [
        b"Loading pretrained model\n",
        b"Fetching 12 files: 0%\n",
        b"Fetching 12 files: 100%\n",
    ]
    captured: list[str] = []
    monkeypatch.setattr(
        gguf_mod.subprocess, "Popen", lambda cmd, **_kw: FakeProc(cmd, fake_lines)
    )

    convert_to_gguf(merged, quant="q8_0", on_progress=captured.append)
    assert captured == [
        "Loading pretrained model",
        "Fetching 12 files: 0%",
        "Fetching 12 files: 100%",
    ]


def test_convert_to_gguf_writes_via_partial_then_renames(tmp_path: Path, monkeypatch):
    """Atomic-write contract: the subprocess receives a .partial path; the
    trainer renames to the final path only after a successful exit. That
    way an interrupted convert leaves a .partial sibling instead of a
    half-written .gguf the cache logic would happily reuse."""
    _stub_convert_script(tmp_path, monkeypatch)
    merged = tmp_path / "merged"
    merged.mkdir()

    seen_outfiles: list[str] = []

    def write_out(cmd):
        out_idx = cmd.index("--outfile") + 1
        seen_outfiles.append(cmd[out_idx])
        Path(cmd[out_idx]).write_bytes(b"\x00")

    monkeypatch.setattr(
        gguf_mod.subprocess, "Popen", _fake_popen_factory([], write_out)
    )
    out = convert_to_gguf(merged, quant="q8_0")

    # Subprocess saw a .partial path…
    assert len(seen_outfiles) == 1
    assert seen_outfiles[0].endswith(".gguf.partial")
    # …but the final return value is the canonical .gguf, and it exists.
    assert out.suffix == ".gguf"
    assert out.exists()
    # The .partial sibling was renamed away, not left behind.
    assert not Path(seen_outfiles[0]).exists()


def test_convert_to_gguf_cleans_up_partial_on_subprocess_failure(tmp_path: Path, monkeypatch):
    """If the convert script crashes mid-write, the .partial file should be
    deleted so the next attempt starts fresh — not silently reuse a corrupt
    cache."""
    _stub_convert_script(tmp_path, monkeypatch)
    merged = tmp_path / "merged"
    merged.mkdir()

    seen_outfiles: list[str] = []

    class FailingProc:
        def __init__(self, cmd):
            self._cmd = cmd
            self.stdout = self
            self._lines = iter([b"some output\n", b""])

        def readline(self):
            return next(self._lines, b"")

        def wait(self):
            # Write a partial file then "crash"
            out_idx = self._cmd.index("--outfile") + 1
            seen_outfiles.append(self._cmd[out_idx])
            Path(self._cmd[out_idx]).write_bytes(b"corrupt")
            return 1

    monkeypatch.setattr(
        gguf_mod.subprocess, "Popen", lambda cmd, **_kw: FailingProc(cmd)
    )

    with pytest.raises(gguf_mod.subprocess.CalledProcessError):
        convert_to_gguf(merged, quant="q8_0")

    # No final .gguf, no .partial — both cleaned up.
    assert not (merged.parent / "merged-q8_0.gguf").exists()
    assert not Path(seen_outfiles[0]).exists()


def test_run_with_progress_raises_with_captured_tail(tmp_path: Path, monkeypatch):
    """Non-zero exit should raise CalledProcessError carrying the recent
    stdout lines, so callers can include them in user-facing error messages."""
    class FakeProc:
        def __init__(self):
            self._lines = iter([b"Loading\n", b"Traceback (most recent)\n", b"Boom\n", b""])
            self.stdout = self

        def readline(self):
            return next(self._lines, b"")

        def wait(self):
            return 1

    monkeypatch.setattr(gguf_mod.subprocess, "Popen", lambda *_a, **_kw: FakeProc())

    with pytest.raises(gguf_mod.subprocess.CalledProcessError) as ei:
        gguf_mod._run_with_progress(["fake-cmd"], None)
    assert "Boom" in (ei.value.output or "")
    assert "Traceback" in (ei.value.output or "")
