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
    pyinstaller --onefile --name "llm-chain-sidecar-$triple" `
        --distpath $out `
        --workpath $build `
        --specpath $build `
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
