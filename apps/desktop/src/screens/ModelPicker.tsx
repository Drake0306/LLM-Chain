import { useEffect, useState } from "react";

import type { ModelEntry } from "../api/client";
import { useApiClient } from "../api/hooks";
import { useSelection } from "../state/selection";
import { loadSettings } from "../state/settings";

function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return n.toLocaleString();
}

export function ModelPicker() {
  const api = useApiClient();
  const { device, model, setModel, technique, setTechnique } = useSelection();
  const [models, setModels] = useState<ModelEntry[] | null>(null);

  const cap =
    technique === "qlora"
      ? device?.capabilities.qlora_max_params
      : device?.capabilities.lora_max_params;
  const includeRestricted = loadSettings().allowRestrictedModels;

  useEffect(() => {
    if (!api) return;
    api.getModels(cap, includeRestricted).then((r) => setModels(r.models));
  }, [api, cap, includeRestricted]);

  if (!device) {
    return (
      <div className="p-6 text-zinc-500">
        Pick a device on the Dashboard first.
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Pick a Model</h1>
          <p className="text-sm text-zinc-500">
            Showing models that fit {device.name} for {technique.toUpperCase()} (≤{" "}
            {cap ? formatParams(cap) : "?"}).
          </p>
        </div>
        <div className="flex gap-2 text-sm">
          {(["qlora", "lora"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTechnique(t)}
              className={`px-3 py-1 rounded-md border ${
                technique === t
                  ? "border-blue-500 bg-blue-50 text-blue-800"
                  : "border-zinc-200 text-zinc-600 hover:border-zinc-400"
              }`}
            >
              {t.toUpperCase()}
            </button>
          ))}
        </div>
      </header>

      {!models && <div className="text-zinc-500">Loading…</div>}
      {models && models.length === 0 && (
        <div className="text-zinc-500">
          No models in the curated allowlist fit this device + technique. Try LoRA, or pick a
          smaller device tier.
        </div>
      )}

      <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {models?.map((m) => {
          const selected = model?.id === m.id;
          return (
            <li key={m.id}>
              <button
                type="button"
                onClick={() => setModel(m)}
                className={`w-full text-left rounded-lg border p-4 transition ${
                  selected
                    ? "border-blue-500 ring-2 ring-blue-200 bg-blue-50"
                    : "border-zinc-200 hover:border-zinc-400"
                }`}
              >
                <div className="flex items-baseline justify-between gap-2">
                  <div className="font-medium">{m.name}</div>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      m.restricted
                        ? "bg-amber-100 text-amber-800 border border-amber-300"
                        : "bg-zinc-100 text-zinc-700"
                    }`}
                  >
                    {m.license}
                  </span>
                </div>
                <div className="text-xs text-zinc-500 font-mono mt-1">{m.id}</div>
                <div className="text-xs text-zinc-600 mt-2">
                  {m.family} • {formatParams(m.params)} params
                </div>
                {m.restricted && m.license_caveat && (
                  <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 mt-2">
                    <span className="font-medium">License caveat:</span>{" "}
                    {m.license_caveat}
                  </div>
                )}
                {m.notes && (
                  <div className="text-xs text-zinc-500 mt-1">{m.notes}</div>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
