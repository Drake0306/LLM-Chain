import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import type { DiffResult, Run } from "../api/client";
import { useApiClient } from "../api/hooks";

export function Diff() {
  const api = useApiClient();
  const [runs, setRuns] = useState<Run[]>([]);
  const [aId, setAId] = useState<string>("");
  const [bId, setBId] = useState<string>("");
  const [result, setResult] = useState<DiffResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!api) return;
    api.listRuns().then((r) => {
      const succeeded = r.runs.filter(
        (x) =>
          x.status === "succeeded" &&
          x.config.purpose !== "lr_finder" &&
          x.config.backend !== "mlx_vlm" &&
          x.config.backend !== "cuda_vlm",
      );
      setRuns(succeeded);
    });
  }, [api]);

  const aRun = runs.find((r) => r.id === aId);
  const bRun = runs.find((r) => r.id === bId);
  const baseMismatch =
    !!aRun && !!bRun && aRun.config.model_id !== bRun.config.model_id;
  const shapeMismatch =
    !!aRun &&
    !!bRun &&
    (aRun.config.lora_rank !== bRun.config.lora_rank ||
      aRun.config.lora_alpha !== bRun.config.lora_alpha);
  const ready = !!aId && !!bId && aId !== bId && !baseMismatch && !shapeMismatch;

  async function handleDiff() {
    if (!api || !ready) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.diffRuns(aId, bId);
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="p-6 max-w-5xl space-y-6">
      <header className="flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Adapter diff</h1>
          <p className="text-sm text-zinc-500 leading-relaxed max-w-3xl">
            Per-layer Frobenius norm of <code>‖A − B‖</code> across two
            LoRA adapters. Higher bars mean the layer changed more
            relative to the other adapter — useful for "where did this
            fine-tune actually move things?" forensics.
          </p>
        </div>
        <Link to="/library" className="text-sm text-zinc-600 hover:text-zinc-900">
          ← Back to Library
        </Link>
      </header>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <RunPicker label="Adapter A" runs={runs} value={aId} onChange={setAId} />
        <RunPicker label="Adapter B" runs={runs} value={bId} onChange={setBId} />
      </section>

      {baseMismatch && (
        <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2">
          Diff needs runs that share a base model.
        </div>
      )}
      {shapeMismatch && !baseMismatch && (
        <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2">
          Diff needs identical LoRA rank/alpha across both runs.
        </div>
      )}

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleDiff}
          disabled={!ready || busy}
          className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm disabled:bg-zinc-300"
        >
          {busy ? "Computing…" : "Compute diff"}
        </button>
        {error && (
          <span className="text-xs text-red-700 leading-relaxed whitespace-pre-wrap">
            {error}
          </span>
        )}
      </div>

      {result && <DiffHeatmap result={result} />}
    </div>
  );
}

function RunPicker({
  label,
  runs,
  value,
  onChange,
}: {
  label: string;
  runs: Run[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-1">
      <label className="block text-sm font-medium">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
      >
        <option value="">— pick —</option>
        {runs.map((r) => (
          <option key={r.id} value={r.id}>
            {r.id} — {r.config.model_id} (r{r.config.lora_rank}/α
            {r.config.lora_alpha})
          </option>
        ))}
      </select>
    </div>
  );
}

function DiffHeatmap({ result }: { result: DiffResult }) {
  const max = result.summary.max_frobenius || 1;
  // Group by layer index for cleaner labels — names look like
  // "base_model.layers.0.q_proj.lora_A.weight"; we extract the
  // numeric index for axis sorting.
  const formatted = useMemo(() => {
    return result.layers.map((l) => {
      const layerMatch = l.key.match(/layers\.(\d+)\./);
      const layerIdx = layerMatch ? parseInt(layerMatch[1], 10) : -1;
      const tail = l.key
        .replace(/^base_model\.layers\.\d+\./, "")
        .replace(/^base_model\./, "");
      return { ...l, layerIdx, tail };
    });
  }, [result]);

  return (
    <section className="space-y-2 border border-zinc-200 rounded-lg bg-white p-4">
      <div className="flex items-baseline justify-between">
        <h2 className="text-sm font-semibold">
          Δ per layer · {result.layers.length} matched
        </h2>
        <span className="text-xs text-zinc-500">
          base: {result.base_model} · max ‖Δ‖ ={" "}
          {result.summary.max_frobenius.toFixed(4)} · mean ={" "}
          {result.summary.mean_frobenius.toFixed(4)}
        </span>
      </div>
      <div className="space-y-1 max-h-[40rem] overflow-y-auto pr-2">
        {formatted.map((l) => {
          const pct = Math.max(0.5, (l.frobenius / max) * 100);
          return (
            <div
              key={l.key}
              className="grid grid-cols-[10rem_1fr_5rem] items-center gap-2 text-xs"
              title={l.key}
            >
              <div className="font-mono truncate text-zinc-600">
                {l.layerIdx >= 0 ? `L${l.layerIdx} ` : ""}
                {l.tail}
              </div>
              <div className="bg-zinc-100 rounded h-3 overflow-hidden">
                <div
                  className="h-full bg-blue-500"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="text-right font-mono text-zinc-700">
                {l.frobenius.toFixed(4)}
              </div>
            </div>
          );
        })}
      </div>
      {(result.summary.unmatched_keys.only_a.length > 0 ||
        result.summary.unmatched_keys.only_b.length > 0) && (
        <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 leading-relaxed">
          Some keys appear in only one adapter and aren't represented above:{" "}
          A-only {result.summary.unmatched_keys.only_a.length} · B-only{" "}
          {result.summary.unmatched_keys.only_b.length}.
        </div>
      )}
    </section>
  );
}
