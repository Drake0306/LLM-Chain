# AMD ROCm on Windows via WSL2 — local validation walkthrough

**Status:** experimental opt-in. `HfRocmTrainer` is a real `HfCudaTrainer` subclass (HIP reuses CUDA's API surface) gated by `LLM_CHAIN_ROCM_EXPERIMENTAL=1`. By default the trainer refuses to instantiate; with the env var set, the LoRA path runs under a loud warning. QLoRA stays refused because `bitsandbytes` is CUDA-only. See [`supported-hardware.md`](supported-hardware.md#amd-rocm-experimental-opt-in) for the high-level framing and [`amd-rocm-quickstart.md`](amd-rocm-quickstart.md) for the parallel native-Linux walkthrough.

The whole point of LLM-Chain is local-only training, so the validation path stays local: your Windows box, your AMD GPU, your WSL2 install. No cloud, no shared infra.

## 1. Pre-flight: is your GPU on the WSL2 ROCm support list?

ROCm-on-WSL2 is **not** universal — AMD ships it for a specific cut of cards on Windows 11. Check yours against AMD's list at <https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html> before going further.

As of early 2026 the consumer cards officially supported are roughly:

- **RDNA4** (gfx1200 / gfx1201): Radeon **RX 9070 / 9070 XT** — added in ROCm 6.3+. **Use ROCm 6.3 or newer**; the 6.2.x bundle linked in step 3 below predates RDNA4 support and will not see your card.
- **RDNA3** (gfx1100 / gfx1101 / gfx1102): Radeon RX 7900 XTX / 7900 XT / 7900 GRE; Radeon Pro W7900 / W7800.
- RDNA2 / RX 6000 series is **not** on the official list — it may still kind-of work via the `HSA_OVERRIDE_GFX_VERSION` env var hack but is unsupported and we won't be able to help if it doesn't.

To find your GPU on Windows: open **Device Manager → Display adapters**, or run `wmic path win32_VideoController get name` in a Windows terminal.

If your card isn't on the list: the probe will probably not see it under WSL2 and you have two options — wait for AMD to widen support, or run native Linux (dual boot or a spare drive) where the supported card list is broader.

> **Your card (RX 9070 / RDNA4) note:** RDNA4 support landed in ROCm 6.3. Make sure step 3 below installs **6.3 or newer**, not 6.2.x. Set `HSA_OVERRIDE_GFX_VERSION=12.0.0` in your shell (`export HSA_OVERRIDE_GFX_VERSION=12.0.0` in `~/.bashrc`) only as a fallback if `rocminfo` reports your card as "unsupported gfx target" with the default install — usually unnecessary on 6.3+ but cheap insurance.

## 2. Windows side: Adrenalin driver + WSL2

You need:
- **Windows 11** (22H2 or newer recommended). Windows 10 is not officially supported for ROCm-on-WSL.
- **Adrenalin Pro driver 24.10.x or newer** (the WSL-aware one). Download from <https://www.amd.com/en/support>; pick "AMD Software: Adrenalin Edition" for your card. Reboot after installing.
- **WSL2 with kernel 5.15+**.

Open **PowerShell as Administrator** and run:

```powershell
wsl --update
wsl --install -d Ubuntu-22.04
```

If WSL was never enabled, the first command may need a reboot. After Ubuntu finishes installing it'll prompt you for a username + password — that's the Linux user inside WSL, **not** your Windows account.

Verify GPU passthrough is working at the kernel level:

```powershell
wsl --status
```

You should see WSL Version 2 and a kernel >= 5.15.

## 3. Inside Ubuntu (WSL): install AMD's WSL ROCm bundle

Open the **Ubuntu-22.04** app from the Start menu. Everything below runs inside that shell.

```bash
sudo apt update
sudo apt install -y wget gnupg2

# Pull AMD's installer for ROCm 6.3+ (RDNA4 / RX 9070 needs 6.3 or newer;
# version numbers change every quarter, double-check at
# https://repo.radeon.com/amdgpu-install/ for the current latest).
wget https://repo.radeon.com/amdgpu-install/6.3/ubuntu/jammy/amdgpu-install_6.3.60300-1_all.deb
sudo apt install -y ./amdgpu-install_6.3.60300-1_all.deb

# The --usecase=wsl flag is the magic — it skips the kernel module install
# (the Windows host driver already provides /dev/dxg) and installs only the
# userspace runtime + HIP libraries.
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms

# Add yourself to the render and video groups so non-root processes can talk
# to the GPU. Then close and reopen the Ubuntu shell for it to take effect.
sudo usermod -a -G render,video $LOGNAME
```

Close and reopen Ubuntu. Verify:

```bash
rocminfo | head -20
rocm-smi
```

`rocminfo` should print an `Agent` block for your GPU (look for "gfx11..." for RDNA3). `rocm-smi` should show GPU temperature, power, VRAM. If both work, the WSL passthrough is solid; if not, see the troubleshooting section at the bottom.

## 4. Install Python 3.11 and clone LLM-Chain inside WSL

LLM-Chain pins Python 3.11.x. Ubuntu 22.04 ships 3.10 by default, so add the deadsnakes PPA:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev git
```

Clone the repo into your **WSL home directory**, not under `/mnt/c/...` — Windows-mounted paths are dramatically slower for `pip install` and `git`:

```bash
cd ~
git clone https://github.com/Drake0306/LLM-Chain.git
cd LLM-Chain

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e './sidecar[dev]'
```

Note: **do not** add the `[cuda]` extra. `bitsandbytes` is CUDA-only and will either fail to install or install the wrong wheel.

## 5. Install PyTorch with the ROCm wheel

This is the part that makes `torch.version.hip` non-empty, which is what our probe keys off:

```bash
pip uninstall -y torch
# Match the major.minor of your installed ROCm. For RX 9070 / RDNA4 you need
# the rocm6.3 (or newer) wheel — pytorch.org/get-started/locally is the
# source of truth for which combinations exist on any given day.
pip install --pre torch --index-url https://download.pytorch.org/whl/rocm6.3
```

Verify:

```bash
python -c "import torch; print('cuda:', torch.version.cuda); print('hip:', torch.version.hip); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

You want to see `cuda: None`, `hip: 6.x.x...`, `available: True`, and your AMD GPU's name. If `available` is `False`, something is wrong with WSL passthrough — go back to `rocm-smi` and `rocminfo`.

## 6. Run the sidecar and confirm the probe sees your GPU

```bash
uvicorn llm_chain_sidecar.main:app --host 127.0.0.1 --port 8000
```

In another WSL shell (or a Windows browser — WSL2 forwards `127.0.0.1`):

```bash
curl localhost:8000/api/hardware | python -m json.tool
```

You should see a device with `"backend": "rocm"`, your GPU name, and `"warning_codes": ["rocm_unverified", ...]`. If you do — the probe path is fully validated end-to-end on real AMD hardware. That's a meaningful milestone in itself; please open an issue at <https://github.com/Drake0306/LLM-Chain/issues> with a paste of the JSON so we can lock in compatibility for your GPU.

## 7. (Optional) Run the Tauri UI from the Windows side

If you want the desktop UI rather than just the API:

- Option A: install the released `.msi` on Windows. The bundled sidecar inside the MSI is the **CUDA build** — it won't see your AMD card. This option is useful only for clicking through the UI shell, not for AMD-specific work.
- Option B: run `npm run tauri dev` **inside WSL** with WSLg (WSL's built-in Wayland display). The Tauri window will pop up on your Windows desktop and the sidecar it spawns will be the ROCm-aware one you installed in step 5. This is the right setup for AMD validation.

```bash
cd ~/LLM-Chain/apps/desktop
npm install
./scripts/build-sidecar.sh --dev
npm run tauri dev
```

(WSLg ships with WSL2 on Windows 11; you don't need an extra X server.)

## 8. Actually training: arm the experimental flag

`HfRocmTrainer` refuses to instantiate by default — we don't ship silent best-effort training on hardware we've never validated. To unlock the LoRA path on your box:

```bash
export LLM_CHAIN_ROCM_EXPERIMENTAL=1
uvicorn llm_chain_sidecar.main:app --host 127.0.0.1 --port 8000
```

(Or add `export LLM_CHAIN_ROCM_EXPERIMENTAL=1` to your `~/.bashrc` so every WSL shell has it.)

When the flag is set:

- The Dashboard's AMD card switches its amber chip from *"experimental — not yet validated on hardware"* to *"experimental ARMED — LoRA only, please report results"* and becomes selectable.
- `HfRocmTrainer.__init__` prints a loud `WARNING` line into the sidecar log every time a run starts.
- LoRA runs go through; **QLoRA still refuses** with a clear message because `bitsandbytes` is CUDA-only. If you've installed an AMD bitsandbytes fork and want to try QLoRA anyway, file an issue.

If you launch the Tauri shell from inside WSL with WSLg (step 7 option B), the env var inherits naturally. If you install the bundled `.msi` on Windows, the env var won't reach the bundled sidecar — that path stays stub-only.

### Smoke test: real LoRA step on a tiny model

The fastest way to *prove* the path end-to-end:

```bash
cd ~/LLM-Chain/sidecar
LLM_CHAIN_ROCM_EXPERIMENTAL=1 pytest -v -m slow tests/trainers/test_hf_cuda.py::test_real_lora_step_on_cuda
```

`torch.cuda.*` calls route to ROCm transparently. The slow test does a one-step LoRA on `hf-internal-testing/tiny-random-LlamaForCausalLM`. If it passes on your box, that's the gating signal we needed to promote `HfRocmTrainer` from "experimental" to "supported". Please file an issue at <https://github.com/Drake0306/LLM-Chain/issues> with:

- Output of `rocminfo | head -30` (so we can see your `gfx` target)
- Output of `python -c "import torch; print(torch.__version__, torch.version.hip)"`
- Pass / fail of the smoke test
- Any traceback if it failed

## Troubleshooting

- **`rocminfo` returns nothing / can't find devices.** WSL2 passthrough is broken. Common causes: stale Adrenalin driver (update to 24.10+), Windows 10 (upgrade to 11), GPU not on the WSL support list (see step 1).
- **`/dev/dxg: permission denied`.** You forgot to add yourself to the `render` and `video` groups, or you didn't restart the WSL shell after running `usermod`. Run `groups` to confirm — you should see both. If still broken, restart WSL from Windows: `wsl --shutdown` in PowerShell, then re-open Ubuntu.
- **`pip install torch ... rocm6.2` resolves to a CUDA wheel.** You forgot `--index-url`, or pip is using a cached CUDA wheel. Run `pip cache remove torch` and try again.
- **Sidecar starts but `curl /api/hardware` shows no rocm device.** Run the `python -c "import torch..."` snippet from step 5; if `torch.version.hip` is None, pip pulled the wrong torch. Reinstall with the index-url flag.
- **Recharts / Tauri UI won't open under WSLg.** Update WSL: `wsl --update`. WSLg is bundled with WSL2 on Win11 22H2+; older versions need WSL upgraded.

## When you finish

Update [`memory/project_status.md`](../) (or its successor) with what worked / didn't, paste the `/api/hardware` JSON into a new GitHub issue tagged `amd-rocm`, and ping me from your Mac with a link. From there we can scope the opt-in flag and start promoting `HfRocmTrainer` toward "real".
