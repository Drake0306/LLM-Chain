import { useEffect, useState } from "react";

import type { HardwareDevice, HardwareReport } from "../api/client";
import { useApiClient } from "../api/hooks";
import { useSelection } from "../state/selection";

function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return n.toLocaleString();
}

function trainableDevices(devices: HardwareDevice[]): HardwareDevice[] {
  return devices.filter((d) => d.backend === "cuda" || d.backend === "mlx");
}

export function Dashboard() {
  const api = useApiClient();
  const [hw, setHw] = useState<HardwareReport | null>(null);
  const { device, setDevice } = useSelection();

  useEffect(() => {
    if (!api) return;
    api.getHardware().then((report) => {
      setHw(report);
      const candidates = trainableDevices(report.devices);
      if (!device && candidates.length > 0) {
        setDevice(candidates[0]);
      }
    });
  }, [api]);

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
          const selectable = d.backend === "cuda" || d.backend === "mlx";
          const selected = device?.name === d.name && device?.backend === d.backend;
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
              {selectable && (
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
