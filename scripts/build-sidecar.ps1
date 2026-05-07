$ErrorActionPreference = "Stop"

# Build the Python sidecar into a relocatable Windows binary using PyInstaller.
# Output: apps/desktop/src-tauri/binaries/llm-chain-sidecar-<triple>.exe

$root = Resolve-Path "$PSScriptRoot/.."
$triple = (rustc -vV | Select-String "host:").ToString().Split()[-1]
$out = "$root/apps/desktop/src-tauri/binaries"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Push-Location "$root/sidecar"
pip install --quiet pyinstaller
pyinstaller --onefile --name "llm-chain-sidecar-$triple.exe" `
    --distpath $out `
    --workpath $env:TEMP/llm-chain-build `
    --specpath $env:TEMP/llm-chain-build `
    -p . llm_chain_sidecar/main.py
Pop-Location
Write-Host "Built: $out/llm-chain-sidecar-$triple.exe"
