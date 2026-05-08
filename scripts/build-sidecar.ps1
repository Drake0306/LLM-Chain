$ErrorActionPreference = "Stop"

# Build the Python sidecar into a relocatable Windows binary using PyInstaller.
# Output: apps/desktop/src-tauri/binaries/llm-chain-sidecar-<triple>.exe

$root = Resolve-Path "$PSScriptRoot/.."
$triple = (rustc -vV | Select-String "host:").ToString().Split()[-1]
$out = "$root/apps/desktop/src-tauri/binaries"
$build = "$root/sidecar/_pyinstaller_build"
New-Item -ItemType Directory -Force -Path $out | Out-Null
New-Item -ItemType Directory -Force -Path $build | Out-Null

Push-Location "$root/sidecar"
try {
    pip install --quiet pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed (exit $LASTEXITCODE)" }

    # IMPORTANT: workpath/specpath must be on the same drive as the source script.
    # On GitHub-hosted Windows runners $env:TEMP lives on C:, but the repo is on D:,
    # and PyInstaller's makespec calls os.path.relpath, which raises across drives.
    # Use a build dir adjacent to the source instead.
    $hf_templates = Join-Path (python -c "import huggingface_hub; print(huggingface_hub.__path__[0])") "templates"
    # --add-data SRC paths are resolved relative to --specpath, not cwd.
    # Use an absolute path for the sidecar's registry data so PyInstaller
    # finds it regardless of where the spec dir lives.
    $models_data = "$root/sidecar/llm_chain_sidecar/models/data"
    pyinstaller --onefile --name "llm-chain-sidecar-$triple" `
        --distpath $out `
        --workpath $build `
        --specpath $build `
        --add-data "${models_data};llm_chain_sidecar/models/data" `
        --add-data "${hf_templates};huggingface_hub/templates" `
        -p . llm_chain_sidecar/main.py
    if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed (exit $LASTEXITCODE)" }
} finally {
    Pop-Location
}

$expected = "$out/llm-chain-sidecar-$triple.exe"
if (-not (Test-Path $expected)) {
    throw "PyInstaller reported success but $expected does not exist."
}
Write-Host "Built: $expected"
