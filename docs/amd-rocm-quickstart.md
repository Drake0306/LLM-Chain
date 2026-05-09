# AMD ROCm on native Linux — quickstart

**Audience:** you have an AMD Radeon GPU on a native Linux box (Fedora / Ubuntu / Mint / Arch / NixOS / openSUSE / RHEL) and want to fine-tune locally on it. Estimated time: 30 minutes if ROCm is already installed; ~1 hour if you're installing ROCm too.

**Status:** the ROCm code path (probe, capability gate, LoRA trainer, DPO, distillation, inference playground) is fully wired in the sidecar. The only thing the bundled `.deb` / `.rpm` we ship today doesn't give you is a ROCm-flavored PyTorch — that's why this quickstart runs the sidecar **from source** with a ROCm wheel until we ship a separate `linux-rocm` build. You'll be able to validate end-to-end on your hardware right now.

**Author's note:** if you do go through this on real hardware, please file an issue with the output of `/api/hardware` and the smoke-test result — that's the gating signal for promoting the ROCm trainer past "experimental" and for shipping a dedicated ROCm `.deb`.

---

## 1. Verify your card and ROCm version

ROCm support for consumer Radeon GPUs is moving fast and version-sensitive. Match yours up front:

| GPU family | Example cards | gfx target | Minimum ROCm |
| --- | --- | --- | --- |
| RDNA 4 | RX 9070, RX 9070 XT | gfx1200 / gfx1201 | **6.3** |
| RDNA 3 | RX 7900 XTX / XT / GRE, W7900, W7800 | gfx1100 / gfx1101 / gfx1102 | 6.0 |
| RDNA 2 | RX 6800 / 6900 | gfx1030 | unsupported officially; works with `HSA_OVERRIDE_GFX_VERSION=10.3.0` on some cards |
| CDNA / Instinct | MI210, MI250, MI300 | gfx908 / gfx90a / gfx940 | 5.x |

Check what's installed:

```bash
rocminfo | head -30   # look for an "Agent" block with your GPU
rocm-smi              # should print temp / power / VRAM
ls /opt/rocm-*        # which ROCm version is on disk
```

If you see your GPU in `rocminfo` and `rocm-smi`, skip to step 3. If `rocminfo` reports nothing or "Agent 1" is only your CPU, jump to step 2.

## 2. Install ROCm (skip if you already have it)

The official source of truth is <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>. The TL;DR per distro family:

### Ubuntu 22.04 / 24.04 / Mint 21+ / Pop!_OS / Debian 12

```bash
sudo apt update
sudo apt install -y wget gnupg2

# Pull AMD's installer (version pinned to the ROCm release you need; check
# https://repo.radeon.com/amdgpu-install/ for the current release).
wget https://repo.radeon.com/amdgpu-install/6.3/ubuntu/jammy/amdgpu-install_6.3.60300-1_all.deb
sudo apt install -y ./amdgpu-install_6.3.60300-1_all.deb

# Install kernel module + ROCm userspace + HIP libs.
sudo amdgpu-install -y --usecase=graphics,rocm

# Group membership: needed so non-root processes can talk to /dev/kfd.
sudo usermod -a -G render,video $LOGNAME
```

Reboot. Then re-run `rocminfo` from step 1.

### Fedora 39+ / RHEL 9 / Rocky 9

```bash
sudo dnf install -y https://repo.radeon.com/amdgpu-install/6.3/rhel/9.4/amdgpu-install-6.3.60300-1.el9.noarch.rpm
sudo amdgpu-install -y --usecase=graphics,rocm
sudo usermod -a -G render,video $LOGNAME
```

Reboot, then `rocminfo`.

### Arch / Manjaro / EndeavourOS

ROCm lives in the official repos:

```bash
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk
sudo usermod -a -G render,video $LOGNAME
```

Reboot, then `rocminfo`.

### NixOS

ROCm packages are in nixpkgs but the install pattern is declarative — outside the scope of this quickstart. See <https://nixos.wiki/wiki/AMD_GPU>; the userspace runtime you need lands in `pkgs.rocmPackages.clr`.

### Other distros

The amdgpu-install installer supports SLES / openSUSE; for everything else build from source or wait for distro packages. AMD's docs are the source of truth.

## 3. Clone LLM-Chain and set up the venv

```bash
# Use your home dir, not a network mount — pip is much faster locally.
cd ~
git clone https://github.com/Drake0306/LLM-Chain.git
cd LLM-Chain

# Python 3.11.x is required (the sidecar pins >=3.11,<3.12). Most distros
# ship 3.10 or 3.12 by default; install 3.11 from your package manager:
#   - Ubuntu/Mint:  sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt install python3.11 python3.11-venv python3.11-dev
#   - Fedora:       sudo dnf install python3.11
#   - Arch:         pacman -S python-pyenv  # then `pyenv install 3.11`

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e './sidecar[dev]'
```

**Important:** do **not** add the `[cuda]` extra. `bitsandbytes` is CUDA-only and the QLoRA path it powers isn't available on ROCm yet. Plain `[dev]` is correct for AMD.

## 4. Install PyTorch with the ROCm wheel

This is the crucial step. The default PyTorch wheel from PyPI is CPU-only on Linux; we need to replace it with the ROCm-flavored wheel that `pip install` won't pick up by itself.

```bash
pip uninstall -y torch
# Match the major.minor of your installed ROCm. RDNA 4 / RX 9070 needs rocm6.3+.
# pytorch.org/get-started/locally is the source of truth for which combinations
# exist on any given day.
pip install --pre torch --index-url https://download.pytorch.org/whl/rocm6.3
```

Verify the wheel is the right flavor and your GPU is reachable:

```bash
python -c "
import torch
print('cuda:    ', torch.version.cuda)
print('hip:     ', torch.version.hip)
print('available:', torch.cuda.is_available())
print('device:  ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"
```

You want:
- `cuda: None` (we replaced the CUDA wheel)
- `hip: 6.3.x` or similar (the ROCm runtime is wired in)
- `available: True`
- `device: <your GPU name>`

If `available` is False, `rocminfo` worked but PyTorch can't see the device — usually a group-membership issue (`groups` should list `render` and `video`, restart the shell after `usermod`) or a ROCm-version mismatch (the wheel's major.minor must match what's on disk under `/opt/rocm-*`).

## 5. Arm the experimental flag and confirm the probe

`HfRocmTrainer` refuses to instantiate by default — we don't ship best-effort training on hardware we haven't validated. Set the env var to opt in:

```bash
export LLM_CHAIN_ROCM_EXPERIMENTAL=1
# Persist across shells:
echo 'export LLM_CHAIN_ROCM_EXPERIMENTAL=1' >> ~/.bashrc
```

Now run the sidecar and check what it sees:

```bash
uvicorn llm_chain_sidecar.main:app --host 127.0.0.1 --port 8000
```

In a second shell:

```bash
curl localhost:8000/api/hardware | python -m json.tool
```

You should see a `"backend": "rocm"` device with your card's name and `"warning_codes"` containing `"rocm_unverified"`. The top-level `"rocm_experimental_armed"` field should be `true`. **That's milestone 1** — the probe path works end-to-end on real AMD hardware. Please paste the JSON into a GitHub issue at <https://github.com/Drake0306/LLM-Chain/issues> with the tag `amd-rocm` so we can lock in compatibility for your card.

## 6. Smoke test — a real LoRA step on a tiny model

The fastest way to prove the training path:

```bash
cd sidecar
LLM_CHAIN_ROCM_EXPERIMENTAL=1 pytest -v -m slow tests/trainers/test_hf_cuda.py::test_real_lora_step_on_cuda
```

`torch.cuda.*` calls route to ROCm transparently when the wheel is HIP-flavored, so the existing CUDA real-training test exercises the ROCm path. The test does one LoRA step on `hf-internal-testing/tiny-random-LlamaForCausalLM` — should finish in under a minute.

If it passes, that's **milestone 3** and the gating signal we needed. Please file an issue with:

- `rocminfo | head -30` output (so we can see your `gfx` target)
- `python -c "import torch; print(torch.__version__, torch.version.hip)"` output
- `pass` or any traceback

If it passes for you, we'll wire up a dedicated `linux-rocm` CI build that ships a `.deb` / `.rpm` with the ROCm-flavored sidecar pre-bundled — and you won't need this quickstart anymore.

## 7. (Optional) Run the full Tauri UI from source

If you want the desktop UI in front of the validated sidecar:

```bash
# Prereqs (one-time):
#   - Rust stable: https://rustup.rs (or your distro's rustup package)
#   - Node 20+: https://nodejs.org or `nvm install 20`
#   - Tauri Linux build deps:
#       Ubuntu/Mint:  sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev libsoup-3.0-dev librsvg2-dev libayatana-appindicator3-dev patchelf
#       Fedora:       sudo dnf install webkit2gtk4.1-devel gtk3-devel libsoup3-devel librsvg2-devel libayatana-appindicator-gtk3-devel patchelf
#       Arch:         sudo pacman -S webkit2gtk-4.1 gtk3 libsoup3 librsvg libayatana-appindicator patchelf

cd apps/desktop
npm install
../../scripts/build-sidecar.sh --dev    # thin wrapper that re-execs the venv'd sidecar
LLM_CHAIN_ROCM_EXPERIMENTAL=1 npm run tauri dev
```

The Tauri window pops up. The Dashboard's AMD card switches its amber chip from *"experimental — not yet validated on hardware"* to *"experimental ARMED — LoRA only, please report results"* and becomes selectable. Pick a model, pick a dataset, hit **Start training** — the run goes through `HfRocmTrainer` instead of the CUDA path. Adapter lands in `~/.llm-chain/runs/<run_id>/`.

That's **milestone 4** — a real end-to-end UI flow on AMD silicon.

## What's not on the AMD path yet

These intentionally fall over with clear messages on ROCm — not sneakily silent:

- **QLoRA** — `bitsandbytes` is CUDA-only. `HfRocmTrainer` raises with a pointer at this. Use plain LoRA in bf16; RDNA 3/4 has native bf16 so quality is comparable.
- **MLX paths** — MLX is Apple-Silicon-only; `mlx_lm` / `mlx_vlm` aren't installed and aren't reachable.
- **VLM (vision-language) training** — currently CUDA-only path. ROCm support is a follow-up.
- **GGUF llama.cpp build** — works on AMD if you build llama.cpp with ROCm yourself; the bundled `scripts/llama-cpp-bootstrap.sh` builds a CPU-only llama.cpp for now. Export to merged HF format works fine; the GGUF k-quant step is what needs the AMD-flavored llama.cpp build.

## Troubleshooting

**`pip install ... rocm6.3` resolves to a CUDA wheel.** You forgot `--index-url`, or pip is using a cached CUDA wheel. `pip cache remove torch` then retry.

**Sidecar starts but `/api/hardware` shows no rocm device.** `python -c "import torch; print(torch.version.hip)"` — if `None`, pip pulled the wrong torch. Reinstall with the index-url flag.

**`/dev/kfd: permission denied`.** You're not in the `render` and `video` groups, or you didn't restart the shell after `usermod`. Run `groups` to confirm both are listed; if not, log out + back in.

**`HSA_OVERRIDE_GFX_VERSION` rabbithole.** Some users set this to force ROCm to treat their card as a supported one. **Try without it first.** If `rocminfo` reports your card explicitly, the override is unnecessary and might mask a different problem. RX 9070 on ROCm 6.3+ does not need it; RDNA 2 cards typically do.

**Out-of-memory on first LoRA step.** The capability gate should catch this in advance, but if it doesn't: drop `per_device_train_batch_size` in the run config, or pick a smaller base model. `rocm-smi` while training shows real-time VRAM use.

**Inference playground says "MLX backend requires macOS".** You picked `mlx` somewhere. Pick `rocm` (or `cpu`) instead.

## Going further

- The full hardware-tier capability table for your card's VRAM is at [`docs/supported-hardware.md`](supported-hardware.md).
- If you're on Windows + WSL2 instead of native Linux, the parallel doc is [`docs/amd-rocm-wsl2-setup.md`](amd-rocm-wsl2-setup.md). The validation milestones are identical.
- For a more general "running from source" walkthrough independent of AMD, see [`setup.md`](../setup.md) at the repo root.
