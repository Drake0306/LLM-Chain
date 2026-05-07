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
from collections import deque
from collections.abc import Callable
from pathlib import Path

# Optional progress callback: receives one stdout line at a time. The route
# layer wires this to the export-gguf.json state file so the UI can show
# what the subprocess is doing (downloading, fusing, converting) instead of
# a frozen "merging…" spinner.
ProgressCb = Callable[[str], None]


def _run_with_progress(cmd: list[str], on_progress: ProgressCb | None) -> None:
    """Run a subprocess, forwarding each stdout line to ``on_progress``.

    Raises ``subprocess.CalledProcessError`` with the captured tail of stdout
    on non-zero exit so callers can surface the real error without losing
    every line that came before.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail: deque[str] = deque(maxlen=60)
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        if not text:
            continue
        tail.append(text)
        if on_progress is not None:
            try:
                on_progress(text)
            except Exception:  # noqa: BLE001 — never let a UI hook kill the export
                pass
    rc = proc.wait()
    if rc != 0:
        detail = "\n".join(tail) if tail else "(no output captured)"
        raise subprocess.CalledProcessError(
            returncode=rc, cmd=cmd, output=detail
        )

# Quants the convert script can emit in a single pass.
_DIRECT_OUTTYPES: frozenset[str] = frozenset({"f32", "f16", "bf16", "q8_0"})
# Common k-quants users select. The full set llama-quantize accepts is larger;
# we expose a curated subset to keep the UI dropdown sane.
_K_QUANTS: frozenset[str] = frozenset({"q4_k_m", "q5_k_m", "q3_k_m"})
SUPPORTED_QUANTS: frozenset[str] = _DIRECT_OUTTYPES | _K_QUANTS

# Backends whose adapter format mlx_lm wrote. peft can't read these; we have
# to use mlx_lm.fuse to produce a merged HF-compatible directory.
_MLX_BACKENDS: frozenset[str] = frozenset({"mlx", "mlx_vlm"})


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
    """Resolve where the trainer saved the adapter for this run.

    Three layouts in the wild:
    - peft saving directly at ``output_dir`` → ``adapter_model.safetensors``
    - HF Trainer checkpoint dirs → ``output_dir/checkpoint-<step>/`` (peft fmt)
    - mlx_lm writing at ``--adapter-path`` → ``adapters.safetensors`` (plural)
    """
    if (run_dir / "adapter_model.safetensors").exists():
        return run_dir
    if (run_dir / "adapters.safetensors").exists():
        return run_dir
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-", 1)[1]) if p.name.split("-", 1)[1].isdigit() else -1,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No adapter or checkpoints found in {run_dir}")
    return checkpoints[-1]


def merge_adapter(
    run_id: str, runs_root: Path, on_progress: ProgressCb | None = None
) -> Path:
    """Merge the trained adapter into the base model and save as a standalone
    HF-compatible directory.

    Backends differ on adapter format and merge tooling:
    - cuda / cpu / cuda_vlm: HF Trainer + peft → use peft.merge_and_unload
    - mlx / mlx_vlm: mlx_lm.lora wrote ``adapters.safetensors`` → use
      mlx_lm.fuse subprocess (peft can't read this format)

    Returns the merged dir path (``<runs_root>/<run_id>/merged``). Idempotent —
    if the merged dir already has a config.json we reuse it.
    """
    run_dir = runs_root / run_id
    merged_dir = run_dir / "merged"
    if (merged_dir / "config.json").exists():
        return merged_dir

    run_data = json.loads((run_dir / "run.json").read_text())
    backend = run_data["config"].get("backend", "")
    model_id = run_data["config"]["model_id"]

    if backend in _MLX_BACKENDS:
        _merge_via_mlx_fuse(run_dir, merged_dir, model_id, on_progress=on_progress)
    else:
        _merge_via_peft(run_dir, merged_dir, model_id)
    return merged_dir


def _merge_via_mlx_fuse(
    run_dir: Path,
    merged_dir: Path,
    model_id: str,
    on_progress: ProgressCb | None = None,
) -> None:
    """Subprocess to mlx_lm.fuse to merge the adapter into the base weights.

    mlx_lm.fuse is bundled with mlx-lm (installed for the MLX trainer path), so
    no extra bootstrap is needed for the merge step on Apple Silicon. Output is
    a fully HF-compatible directory that the downstream GGUF convert script
    can read. Stdout (model fetch progress, conversion lines) is forwarded to
    ``on_progress`` so the UI can show what the subprocess is doing.
    """
    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", model_id,
        "--adapter-path", str(run_dir),
        "--save-path", str(merged_dir),
    ]
    _run_with_progress(cmd, on_progress)


def _merge_via_peft(run_dir: Path, merged_dir: Path, model_id: str) -> None:
    """In-process peft merge for HF-trained adapters."""
    # Lazy import: pulling torch/transformers at module import would slow down
    # every API request, including ones that never touch exports.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id)
    adapter_dir = find_latest_adapter(run_dir)
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)


def convert_to_gguf(
    merged_dir: Path, quant: str = "q4_k_m", on_progress: ProgressCb | None = None
) -> Path:
    """Emit a ``.gguf`` next to ``merged_dir`` at the requested quant.

    For ``f32/f16/bf16/q8_0`` we hand the quant to convert_hf_to_gguf.py via
    ``--outtype``. For k-quants we first produce an f16 GGUF and then run
    ``llama-quantize`` against it. Each subprocess's stdout is forwarded to
    ``on_progress`` so the UI can show what's happening.
    """
    if quant not in SUPPORTED_QUANTS:
        raise ValueError(
            f"Unsupported quant: {quant}. Supported: {sorted(SUPPORTED_QUANTS)}."
        )

    convert = _find_convert_script()
    out_dir = merged_dir.parent

    if quant in _DIRECT_OUTTYPES:
        out = out_dir / f"{merged_dir.name}-{quant}.gguf"
        _run_with_progress(
            [
                sys.executable,
                str(convert),
                str(merged_dir),
                "--outfile",
                str(out),
                "--outtype",
                quant,
            ],
            on_progress,
        )
        return out

    # k-quant: convert → f16, then llama-quantize → target
    f16_path = out_dir / f"{merged_dir.name}-f16.gguf"
    if not f16_path.exists():
        _run_with_progress(
            [
                sys.executable,
                str(convert),
                str(merged_dir),
                "--outfile",
                str(f16_path),
                "--outtype",
                "f16",
            ],
            on_progress,
        )

    quantize = _find_quantize_binary()
    out = out_dir / f"{merged_dir.name}-{quant}.gguf"
    _run_with_progress([str(quantize), str(f16_path), str(out), quant], on_progress)
    return out
