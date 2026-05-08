#!/usr/bin/env bash
set -euo pipefail

# Build the Python sidecar for use as a Tauri externalBin.
#
# Default: a relocatable PyInstaller binary suitable for shipping in a DMG / MSI.
#   ./scripts/build-sidecar.sh
#
# --dev: a thin shell wrapper around the local .venv. Fast (~1s vs ~10min) but
#        the resulting binary is NOT portable — only good for `npm run tauri dev`
#        on the same machine that owns the venv. Used by CI / release? No.
#   ./scripts/build-sidecar.sh --dev
#
# Output: apps/desktop/src-tauri/binaries/llm-chain-sidecar-<rust-target-triple>

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRIPLE="$(rustc -vV | awk '/^host:/ {print $2}')"
OUT="$ROOT/apps/desktop/src-tauri/binaries"
mkdir -p "$OUT"
TARGET="$OUT/llm-chain-sidecar-${TRIPLE}"

if [[ "${1:-}" == "--dev" ]]; then
    VENV_PY="$ROOT/.venv/bin/python"
    if [[ ! -x "$VENV_PY" ]]; then
        echo "error: $VENV_PY not found. Create the venv and \`pip install -e ./sidecar\` first." >&2
        exit 1
    fi
    cat > "$TARGET" <<EOF
#!/usr/bin/env bash
# Dev-mode sidecar wrapper. Replace via \`./scripts/build-sidecar.sh\` for release.
exec "$VENV_PY" -m llm_chain_sidecar.main "\$@"
EOF
    chmod +x "$TARGET"
    echo "Wrote dev wrapper: $TARGET"
    exit 0
fi

cd "$ROOT/sidecar"
pip install --quiet pyinstaller
HF_TEMPLATES=$(python -c "import huggingface_hub; print(huggingface_hub.__path__[0])")/templates
# --add-data SRC paths are resolved relative to --specpath, not cwd.
# We point specpath at /tmp/llm-chain-build, so a relative source like
# "llm_chain_sidecar/models/data" gets looked up under /tmp/... and
# fails with "Unable to find ...". Use an absolute path — same
# pattern as HF_TEMPLATES above.
MODELS_DATA="$ROOT/sidecar/llm_chain_sidecar/models/data"
# F-B6: the curated.yaml manifest must travel with the package — the
# sidecar reads it at request time via importlib.resources, which
# expects the file under llm_chain_sidecar/datasets/ in the bundle.
CURATED_YAML="$ROOT/sidecar/llm_chain_sidecar/datasets/curated.yaml"
pyinstaller --onefile --name "llm-chain-sidecar-${TRIPLE}" \
    --distpath "$OUT" \
    --workpath /tmp/llm-chain-build \
    --specpath /tmp/llm-chain-build \
    --add-data "${MODELS_DATA}:llm_chain_sidecar/models/data" \
    --add-data "${CURATED_YAML}:llm_chain_sidecar/datasets" \
    --add-data "${HF_TEMPLATES}:huggingface_hub/templates" \
    -p . llm_chain_sidecar/main.py
echo "Built: $TARGET"
