"""GGUF export pipeline: merge LoRA adapter into base, then convert to GGUF.

Uses llama.cpp's `convert_hf_to_gguf.py` as a subprocess for the HF→GGUF step.
For k-quants (q4_k_m, q5_k_m, …) the convert script can't emit them directly,
so we first produce f16 and then run `llama-quantize`. Both the convert script
and the quantize binary are provisioned by `scripts/llama-cpp-bootstrap.sh`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Quants the convert script can emit in a single pass.
_DIRECT_OUTTYPES: frozenset[str] = frozenset({"f32", "f16", "bf16", "q8_0"})
# Common k-quants users select. The full set llama-quantize accepts is larger;
# we expose a curated subset to keep the UI dropdown sane.
_K_QUANTS: frozenset[str] = frozenset({"q4_k_m", "q5_k_m", "q3_k_m"})
SUPPORTED_QUANTS: frozenset[str] = _DIRECT_OUTTYPES | _K_QUANTS


def _llama_cpp_dir() -> Path:
    return Path(
        os.environ.get(
            "LLAMA_CPP_DIR", str(Path.home() / ".llm-chain" / "llama.cpp")
        )
    )


def _find_convert_script() -> Path:
    p = _llama_cpp_dir() / "convert_hf_to_gguf.py"
    if not p.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {p}. "
            "Run scripts/llama-cpp-bootstrap.sh first."
        )
    return p


def _find_quantize_binary() -> Path:
    base = _llama_cpp_dir()
    for c in (base / "build" / "bin" / "llama-quantize", base / "build" / "llama-quantize"):
        if c.exists():
            return c
    raise FileNotFoundError(
        "llama-quantize not built. Run scripts/llama-cpp-bootstrap.sh first."
    )


def find_latest_adapter(run_dir: Path) -> Path:
    """Resolve where peft saved the adapter for this run.

    HF Trainer writes checkpoints under ``output_dir/checkpoint-<step>/``. The
    adapter weights live as ``adapter_model.safetensors`` at the
    highest-numbered checkpoint. Some training paths save the adapter directly
    at output_dir; handle that case too.
    """
    if (run_dir / "adapter_model.safetensors").exists():
        return run_dir
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-", 1)[1]) if p.name.split("-", 1)[1].isdigit() else -1,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No adapter or checkpoints found in {run_dir}")
    return checkpoints[-1]


def merge_adapter(run_id: str, runs_root: Path) -> Path:
    """Merge a peft adapter into base weights and save as a standalone HF dir.

    Returns the merged dir path (``<runs_root>/<run_id>/merged``). Idempotent —
    if the merged dir already has a config.json we reuse it.
    """
    run_dir = runs_root / run_id
    merged_dir = run_dir / "merged"
    if (merged_dir / "config.json").exists():
        return merged_dir

    # Lazy import: pulling torch/transformers at module import would slow down
    # every API request, including ones that never touch exports.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_data = json.loads((run_dir / "run.json").read_text())
    model_id = run_data["config"]["model_id"]

    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id)
    adapter_dir = find_latest_adapter(run_dir)
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    return merged_dir


def convert_to_gguf(merged_dir: Path, quant: str = "q4_k_m") -> Path:
    """Emit a ``.gguf`` next to ``merged_dir`` at the requested quant.

    For ``f32/f16/bf16/q8_0`` we hand the quant to convert_hf_to_gguf.py via
    ``--outtype``. For k-quants we first produce an f16 GGUF and then run
    ``llama-quantize`` against it.
    """
    if quant not in SUPPORTED_QUANTS:
        raise ValueError(
            f"Unsupported quant: {quant}. Supported: {sorted(SUPPORTED_QUANTS)}."
        )

    convert = _find_convert_script()
    out_dir = merged_dir.parent

    if quant in _DIRECT_OUTTYPES:
        out = out_dir / f"{merged_dir.name}-{quant}.gguf"
        subprocess.run(
            [
                sys.executable,
                str(convert),
                str(merged_dir),
                "--outfile",
                str(out),
                "--outtype",
                quant,
            ],
            check=True,
        )
        return out

    # k-quant: convert → f16, then llama-quantize → target
    f16_path = out_dir / f"{merged_dir.name}-f16.gguf"
    if not f16_path.exists():
        subprocess.run(
            [
                sys.executable,
                str(convert),
                str(merged_dir),
                "--outfile",
                str(f16_path),
                "--outtype",
                "f16",
            ],
            check=True,
        )

    quantize = _find_quantize_binary()
    out = out_dir / f"{merged_dir.name}-{quant}.gguf"
    subprocess.run([str(quantize), str(f16_path), str(out), quant], check=True)
    return out
