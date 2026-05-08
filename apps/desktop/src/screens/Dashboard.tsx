import { useEffect, useState } from "react";

import type { HardwareDevice, HardwareReport } from "../api/client";
import { useSidecarStatus } from "../api/hooks";
import { useSelection } from "../state/selection";
import { loadSettings } from "../state/settings";

function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return n.toLocaleString();
}

function isTrainable(d: HardwareDevice, rocmArmed: boolean): boolean {
  if (d.backend === "cuda" || d.backend === "mlx") return true;
  // CPU is selectable when the sidecar reports a non-zero cpu_max_params,
  // i.e. the v1.1 fallback path is wired up. Older sidecars (or hosts where
  // the CPU is somehow disqualified) leave it at 0.
  if (d.backend === "cpu") return d.capabilities.cpu_max_params > 0;
  // ROCm is detected and shown so AMD users see they're recognised. The
  // trainer is unvalidated by default, but users on real AMD hardware can
  // arm the experimental path with LLM_CHAIN_ROCM_EXPERIMENTAL=1 — in that
  // case the sidecar reports rocm_experimental_armed and the card becomes
  // selectable for LoRA runs (QLoRA still refuses; bitsandbytes is CUDA-only).
  if (d.backend === "rocm") return rocmArmed;
  return false;
}

function trainableDevices(devices: HardwareDevice[], rocmArmed: boolean): HardwareDevice[] {
  return devices.filter((d) => isTrainable(d, rocmArmed));
}

export function Dashboard() {
  const { client: api, slow: sidecarSlow, phase } = useSidecarStatus();
  const [hw, setHw] = useState<HardwareReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { device, setDevice } = useSelection();

  useEffect(() => {
    if (!api) return;
    api.getHardware()
      .then((report) => {
        setError(null);
        setHw(report);
        const candidates = trainableDevices(report.devices, !!report.rocm_experimental_armed);
        if (!device && candidates.length > 0) {
          const pref = loadSettings().defaultBackend;
          const preferred =
            pref !== "auto"
              ? candidates.find((d) => d.backend === pref)
              : undefined;
          setDevice(preferred ?? candidates[0]);
        }
      })
      .catch((e: unknown) => setError(String(e)));
  }, [api]);

  if (phase === "dead") {
    return (
      <div className="p-6 space-y-3 max-w-prose">
        <h1 className="text-2xl font-semibold text-red-700">Sidecar stopped</h1>
        <p className="text-sm text-zinc-700 leading-relaxed">
          The Python sidecar process exited. Any in-flight runs were
          terminated; their state on disk is preserved and visible on the
          Runs page once the sidecar is back up. Restart the app to bring
          it back online.
        </p>
        <p className="text-xs text-zinc-500">
          If this happens repeatedly, check the terminal where you launched{" "}
          <code className="font-mono">npm run tauri dev</code> for the
          sidecar's last log lines — they usually point at the underlying
          cause (CUDA driver crash, OOM, missing dependency).
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 space-y-2">
        <h1 className="text-2xl font-semibold">Couldn't reach the sidecar</h1>
        <p className="text-sm text-zinc-600">
          The Python sidecar is running but the UI couldn't fetch /api/hardware. Check the terminal where you ran <code>npm run tauri dev</code> for sidecar errors.
        </p>
        <pre className="text-xs bg-red-50 border border-red-200 rounded p-3 whitespace-pre-wrap">{error}</pre>
      </div>
    );
  }

  if (!api) {
    return (
      <div className="p-6 space-y-2 text-zinc-500">
        <div>Starting sidecar…</div>
        {sidecarSlow && (
          <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-3 max-w-prose">
            The Python sidecar is taking longer than expected to come up.
            Check the terminal where you ran <code>npm run tauri dev</code> for
            errors, or restart the app. We'll keep retrying in the background.
          </p>
        )}
      </div>
    );
  }
  if (!hw) {
    return <div className="p-6 text-zinc-500">Probing hardware…</div>;
  }

  return (
    <div className="p-6 space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Your Machine</h1>
        <p className="text-sm text-zinc-500">
          {hw.os} {hw.os_version} • {hw.cpu.cores} cores • {hw.system_ram_gb} GB RAM
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {hw.devices.map((d, i) => {
          const rocmArmed = !!hw.rocm_experimental_armed;
          const selectable = isTrainable(d, rocmArmed);
          const selected = device?.name === d.name && device?.backend === d.backend;
          const isCpu = d.backend === "cpu";
          const isRocm = d.backend === "rocm";
          return (
            <button
              key={i}
              type="button"
              onClick={() => selectable && setDevice(d)}
              disabled={!selectable}
              className={`text-left rounded-lg border p-4 transition ${
                selected
                  ? "border-blue-500 ring-2 ring-blue-200 bg-blue-50"
                  : selectable
                  ? "border-zinc-200 hover:border-zinc-400"
                  : "border-zinc-200 opacity-60 cursor-not-allowed"
              }`}
            >
              <div className="flex items-baseline justify-between gap-2">
                <div className="font-medium">{d.name}</div>
                <span className="text-xs uppercase tracking-wide text-zinc-500">{d.backend}</span>
              </div>
              {isRocm && (
                <div className="mt-2 inline-flex items-center gap-1 rounded-full bg-amber-50 border border-amber-200 px-2 py-0.5 text-xs font-medium text-amber-800">
                  {rocmArmed
                    ? "experimental ARMED — LoRA only, please report results"
                    : "experimental — not yet validated on hardware"}
                </div>
              )}
              <div className="mt-2 text-sm">
                {d.vram_gb > 0
                  ? `${d.vram_gb} GB ${d.memory_kind}`
                  : "no GPU memory (CPU)"}
              </div>
              {isCpu && d.capabilities.cpu_max_params > 0 && (
                <dl className="mt-3 grid grid-cols-1 gap-2 text-xs text-zinc-600">
                  <div>
                    <dt className="uppercase tracking-wide text-zinc-400">CPU LoRA cap</dt>
                    <dd>{formatParams(d.capabilities.cpu_max_params)} params</dd>
                  </div>
                </dl>
              )}
              {!isCpu && d.capabilities.qlora_max_params > 0 && (
                <dl className="mt-3 grid grid-cols-3 gap-2 text-xs text-zinc-600">
                  <div>
                    <dt className="uppercase tracking-wide text-zinc-400">QLoRA</dt>
                    <dd>{formatParams(d.capabilities.qlora_max_params)}</dd>
                  </div>
                  <div>
                    <dt className="uppercase tracking-wide text-zinc-400">LoRA</dt>
                    <dd>{formatParams(d.capabilities.lora_max_params)}</dd>
                  </div>
                  <div>
                    <dt className="uppercase tracking-wide text-zinc-400">Full FT</dt>
                    <dd>{formatParams(d.capabilities.full_ft_max_params)}</dd>
                  </div>
                </dl>
              )}
              {d.capabilities.warning_codes.length > 0 && (
                <div className="mt-2 text-xs text-amber-700">
                  {d.capabilities.notes}
                </div>
              )}
            </button>
          );
        })}
      </div>

      <details className="text-sm text-zinc-600 rounded-md border border-zinc-200 p-3">
        <summary className="font-medium cursor-pointer">
          What do QLoRA / LoRA / Full FT mean?
        </summary>
        <dl className="mt-3 space-y-2 text-xs leading-relaxed">
          <div>
            <dt className="font-medium text-zinc-700">QLoRA</dt>
            <dd>
              Base model is quantized to 4-bit, then LoRA adapters are trained on
              top. Biggest model that fits a given device. Slight quality hit vs
              LoRA, much smaller VRAM footprint. Recommended for consumer GPUs and
              Apple Silicon.
            </dd>
          </div>
          <div>
            <dt className="font-medium text-zinc-700">LoRA</dt>
            <dd>
              Base model stays in full precision (bf16 / fp16); only the small
              adapter weights train. Slightly faster steps and slightly higher
              quality than QLoRA, but the base eats more VRAM so the size cap is
              lower.
            </dd>
          </div>
          <div>
            <dt className="font-medium text-zinc-700">Full fine-tune (Full FT)</dt>
            <dd>
              Every parameter of the base model trains. Highest quality but
              expensive — the cap is roughly 1/20th of QLoRA. Mostly useful for
              very small models or when LoRA quality isn't enough.
            </dd>
          </div>
          <div>
            <dt className="font-medium text-zinc-700">Memory kind</dt>
            <dd>
              <span className="font-mono">dedicated</span> = standalone GPU VRAM
              (NVIDIA / AMD).{" "}
              <span className="font-mono">unified</span> = Apple Silicon, where
              the GPU shares the system RAM pool at full speed (we count 75% of
              it).{" "}
              <span className="font-mono">shared</span> = Windows DDR-over-PCIe
              pseudo-VRAM; treated as effectively zero because it's ~20× slower
              than real VRAM.
            </dd>
          </div>
        </dl>
      </details>

      {device && (
        <p className="text-sm text-zinc-600">
          Training will run on <span className="font-medium">{device.name}</span> ({device.backend}).
        </p>
      )}
    </div>
  );
}
