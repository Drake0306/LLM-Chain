import { useEffect, useState } from "react";

import type { HardwareDevice, HardwareReport } from "../api/client";
import { useApiClient } from "../api/hooks";
import { useSelection } from "../state/selection";
import { loadSettings } from "../state/settings";

function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return n.toLocaleString();
}

function isTrainable(d: HardwareDevice): boolean {
  if (d.backend === "cuda" || d.backend === "mlx") return true;
  // CPU is selectable when the sidecar reports a non-zero cpu_max_params,
  // i.e. the v1.1 fallback path is wired up. Older sidecars (or hosts where
  // the CPU is somehow disqualified) leave it at 0.
  return d.backend === "cpu" && d.capabilities.cpu_max_params > 0;
}

function trainableDevices(devices: HardwareDevice[]): HardwareDevice[] {
  return devices.filter(isTrainable);
}

export function Dashboard() {
  const api = useApiClient();
  const [hw, setHw] = useState<HardwareReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { device, setDevice } = useSelection();

  useEffect(() => {
    if (!api) return;
    api.getHardware()
      .then((report) => {
        setError(null);
        setHw(report);
        const candidates = trainableDevices(report.devices);
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

  if (!api || !hw) {
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
          const selectable = isTrainable(d);
          const selected = device?.name === d.name && device?.backend === d.backend;
          const isCpu = d.backend === "cpu";
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
              <div className="flex items-baseline justify-between">
                <div className="font-medium">{d.name}</div>
                <span className="text-xs uppercase tracking-wide text-zinc-500">{d.backend}</span>
              </div>
              <div className="mt-2 text-sm">
                {d.vram_gb > 0
                  ? `${d.vram_gb} GB ${d.memory_kind}`
                  : "no GPU memory (CPU)"}
              </div>
              {selectable && isCpu && (
                <dl className="mt-3 grid grid-cols-1 gap-2 text-xs text-zinc-600">
                  <div>
                    <dt className="uppercase tracking-wide text-zinc-400">CPU LoRA cap</dt>
                    <dd>{formatParams(d.capabilities.cpu_max_params)} params</dd>
                  </div>
                </dl>
              )}
              {selectable && !isCpu && (
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

      {device && (
        <p className="text-sm text-zinc-600">
          Training will run on <span className="font-medium">{device.name}</span> ({device.backend}).
        </p>
      )}
    </div>
  );
}
