#!/usr/bin/env bash
# Bootstraps the llama.cpp tooling LLM-Chain needs for GGUF export.
#
# - pip-installs the `gguf` package (used by convert_hf_to_gguf.py).
# - Clones llama.cpp into ~/.llm-chain/llama.cpp (or $LLAMA_CPP_DIR).
# - Builds only the `llama-quantize` target — that's the one the export
#   pipeline shells out to for k-quant levels (q4_k_m etc.). The convert
#   script alone covers f16/q8_0/bf16/f32, so quantize is optional but
#   recommended.
#
# Idempotent — re-running pulls the latest llama.cpp main and rebuilds.
set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/.llm-chain/llama.cpp}"
PYTHON="${PYTHON:-python3}"

echo "==> Installing gguf python package"
"$PYTHON" -m pip install --upgrade "gguf>=0.9"

if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  echo "==> Cloning llama.cpp into $LLAMA_CPP_DIR"
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
else
  echo "==> Updating llama.cpp at $LLAMA_CPP_DIR"
  git -C "$LLAMA_CPP_DIR" pull --ff-only
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "WARN: cmake not found. Skipping llama-quantize build."
  echo "      f16/q8_0 export will work via convert_hf_to_gguf.py;"
  echo "      k-quants (q4_k_m etc.) require building llama-quantize."
  exit 0
fi

echo "==> Building llama-quantize"
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize -j

echo "==> Done. GGUF export tooling ready at $LLAMA_CPP_DIR"
