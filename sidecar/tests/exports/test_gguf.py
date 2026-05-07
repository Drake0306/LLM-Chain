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


def test_convert_to_gguf_direct_outtype_runs_one_subprocess(tmp_path: Path, monkeypatch):
    _stub_convert_script(tmp_path, monkeypatch)
    merged = tmp_path / "merged"
    merged.mkdir()

    calls = []
    def fake_run(cmd, check):
        calls.append(cmd)
        # Touch the expected output file so downstream callers can stat it.
        out_idx = cmd.index("--outfile") + 1
        Path(cmd[out_idx]).write_bytes(b"\x00")
        return MagicMock(returncode=0)

    monkeypatch.setattr(gguf_mod.subprocess, "run", fake_run)
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

    calls = []
    def fake_run(cmd, check):
        calls.append(cmd)
        # Mimic both subprocess outputs by creating the expected file.
        if "--outfile" in cmd:
            out = Path(cmd[cmd.index("--outfile") + 1])
        else:
            out = Path(cmd[2])
        out.write_bytes(b"\x00")
        return MagicMock(returncode=0)

    monkeypatch.setattr(gguf_mod.subprocess, "run", fake_run)
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

    calls = []
    def fake_run(cmd, check):
        calls.append(cmd)
        if "--outfile" not in cmd:
            Path(cmd[2]).write_bytes(b"\x00")
        return MagicMock(returncode=0)

    monkeypatch.setattr(gguf_mod.subprocess, "run", fake_run)
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
