import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import type { MergeAdapterEntry, MergeMethod, Run } from "../api/client";
import { useApiClient } from "../api/hooks";

const METHODS: { value: MergeMethod; label: string; help: string }[] = [
  {
    value: "linear",
    label: "Linear",
    help: "Element-wise weighted average. Simplest and most predictable; start here.",
  },
  {
    value: "ties",
    label: "TIES",
    help: "Sign-aware trim + redundancy resolution. Less prone to averaging-out complementary capabilities than plain linear.",
  },
  {
    value: "dare",
    label: "DARE",
    help: "Random sparsification then weighted sum. Cheap and often surprisingly good.",
  },
];

export function Merge() {
  const api = useApiClient();
  const navigate = useNavigate();

  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<Map<string, number>>(new Map());
  const [method, setMethod] = useState<MergeMethod>("linear");
  const [tiesDensity, setTiesDensity] = useState(0.2);
  const [dareDropP, setDareDropP] = useState(0.5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultRunId, setResultRunId] = useState<string | null>(null);

  useEffect(() => {
    if (!api) return;
    api.listRuns().then((r) => {
      const succeeded = r.runs.filter(
        (x) =>
          x.status === "succeeded" &&
          // Skip LR-finder sweeps and merged runs themselves —
          // merging a merge is fine in theory but adds confusion.
          x.config.purpose !== "lr_finder" &&
          x.config.purpose !== "merged",
      );
      setRuns(succeeded);
    });
  }, [api]);

  // Group by base model id; merging needs same-base inputs.
  const runsByBase = useMemo(() => {
    const m = new Map<string, Run[]>();
    for (const r of runs) {
      const list = m.get(r.config.model_id) ?? [];
      list.push(r);
      m.set(r.config.model_id, list);
    }
    return m;
  }, [runs]);

  // Once one is selected, only same-base + same rank/alpha siblings
  // stay selectable. Surface the lock visually so the user understands
  // why other groups grey out.
  const lock = useMemo(() => {
    if (selected.size === 0) return null;
    const first = runs.find((r) => selected.has(r.id));
    if (!first) return null;
    return {
      model_id: first.config.model_id,
      rank: first.config.lora_rank,
      alpha: first.config.lora_alpha,
    };
  }, [selected, runs]);

  function toggle(run: Run) {
    setSelected((prev) => {
      const next = new Map(prev);
      if (next.has(run.id)) {
        next.delete(run.id);
      } else {
        next.set(run.id, 1.0);
      }
      return next;
    });
  }

  function setWeight(runId: string, w: number) {
    setSelected((prev) => {
      const next = new Map(prev);
      if (next.has(runId)) next.set(runId, w);
      return next;
    });
  }

  async function handleMerge() {
    if (!api) return;
    if (selected.size < 2) {
      setError("Pick at least two adapters to merge.");
      return;
    }
    setBusy(true);
    setError(null);
    setResultRunId(null);
    try {
      const entries: MergeAdapterEntry[] = Array.from(selected.entries()).map(
        ([run_id, weight]) => ({ run_id, weight }),
      );
      const opts: Record<string, number> = {};
      if (method === "ties") opts.density = tiesDensity;
      if (method === "dare") opts.drop_p = dareDropP;
      const result = await api.mergeRuns({
        runs: entries,
        method,
        method_options: opts,
      });
      setResultRunId(result.id);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  }

  const selectedCount = selected.size;
  const ready = selectedCount >= 2;

  return (
    <div className="p-6 max-w-5xl space-y-6">
      <header className="flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Merge adapters</h1>
          <p className="text-sm text-zinc-500 leading-relaxed max-w-3xl">
            Combine 2–8 LoRA adapters that share a base model + rank/alpha
            into a new adapter. The result lands in your Library tagged
            <code className="text-xs bg-zinc-100 px-1 rounded mx-1">
              purpose=merged
            </code>
            and works in the playground / eval / GGUF export like any other
            run.
          </p>
        </div>
        <Link to="/library" className="text-sm text-zinc-600 hover:text-zinc-900">
          ← Back to Library
        </Link>
      </header>

      {runs.length < 2 && (
        <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2">
          Need at least 2 succeeded non-merged adapters in your library.
        </p>
      )}

      <div className="space-y-4">
        {Array.from(runsByBase.entries()).map(([modelId, group]) => {
          const groupCompatible =
            !lock ||
            (lock.model_id === modelId &&
              group.some(
                (r) =>
                  r.config.lora_rank === lock.rank &&
                  r.config.lora_alpha === lock.alpha,
              ));
          return (
            <div key={modelId} className="space-y-2">
              <h2 className="text-sm font-semibold text-zinc-700">
                {modelId}
                {!groupCompatible && (
                  <span className="ml-2 text-xs font-normal text-zinc-500">
                    different base model than your selection
                  </span>
                )}
              </h2>
              <div className="space-y-1">
                {group.map((r) => {
                  const isPicked = selected.has(r.id);
                  const shapeMismatch =
                    !!lock &&
                    (r.config.lora_rank !== lock.rank ||
                      r.config.lora_alpha !== lock.alpha);
                  const disabled =
                    !isPicked &&
                    (!groupCompatible || shapeMismatch || selected.size >= 8);
                  return (
                    <div
                      key={r.id}
                      className={`flex items-center gap-3 px-3 py-2 rounded border ${
                        isPicked
                          ? "border-blue-300 bg-blue-50"
                          : "border-zinc-200 bg-white"
                      } ${disabled ? "opacity-50" : ""}`}
                    >
                      <input
                        type="checkbox"
                        checked={isPicked}
                        disabled={disabled}
                        onChange={() => toggle(r)}
                      />
                      <div className="flex-1 min-w-0">
                        <div className="font-mono text-xs text-zinc-500">
                          {r.id}
                        </div>
                        <div className="text-xs text-zinc-500">
                          rank {r.config.lora_rank} · α{" "}
                          {r.config.lora_alpha}
                        </div>
                      </div>
                      {isPicked && (
                        <input
                          type="number"
                          step={0.1}
                          min={0}
                          max={100}
                          value={selected.get(r.id) ?? 1.0}
                          onChange={(e) => setWeight(r.id, +e.target.value)}
                          className="w-20 rounded border border-blue-300 px-2 py-0.5 text-sm"
                          title="weight"
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      <section className="rounded-lg border border-zinc-200 p-4 space-y-3">
        <h2 className="text-sm font-medium">Merge method</h2>
        <div className="flex gap-2 text-sm">
          {METHODS.map((m) => (
            <button
              key={m.value}
              type="button"
              onClick={() => setMethod(m.value)}
              className={`px-3 py-1.5 rounded border ${
                method === m.value
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white border-zinc-300 hover:bg-zinc-50"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        <p className="text-xs text-zinc-600">
          {METHODS.find((m) => m.value === method)?.help}
        </p>
        {method === "ties" && (
          <label className="flex items-center gap-2 text-xs text-zinc-700">
            density
            <input
              type="number"
              step={0.05}
              min={0.05}
              max={1}
              value={tiesDensity}
              onChange={(e) => setTiesDensity(+e.target.value)}
              className="w-20 rounded border border-zinc-300 px-2 py-0.5"
            />
            <span className="text-zinc-500">
              fraction of largest |Δ| to keep per adapter
            </span>
          </label>
        )}
        {method === "dare" && (
          <label className="flex items-center gap-2 text-xs text-zinc-700">
            drop_p
            <input
              type="number"
              step={0.05}
              min={0}
              max={0.95}
              value={dareDropP}
              onChange={(e) => setDareDropP(+e.target.value)}
              className="w-20 rounded border border-zinc-300 px-2 py-0.5"
            />
            <span className="text-zinc-500">
              fraction of elements to drop per adapter (rescaled)
            </span>
          </label>
        )}
      </section>

      <div className="flex items-center gap-3 pt-3 border-t border-zinc-200">
        <button
          type="button"
          onClick={handleMerge}
          disabled={busy || !ready}
          className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm disabled:bg-zinc-300"
        >
          {busy ? "Merging…" : `Merge ${selectedCount} adapters`}
        </button>
        {error && (
          <span className="text-xs text-red-700 leading-relaxed whitespace-pre-wrap">
            {error}
          </span>
        )}
      </div>

      {resultRunId && (
        <div className="rounded-md border border-emerald-200 bg-emerald-50 p-3 space-y-2">
          <div className="text-sm font-medium text-emerald-900">
            Merged run created.
          </div>
          <div className="text-xs font-mono text-emerald-900 break-all">
            {resultRunId}
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => navigate(`/runs/${resultRunId}`)}
              className="rounded-md bg-emerald-600 text-white px-3 py-1.5 text-sm hover:bg-emerald-700"
            >
              Open run →
            </button>
            <button
              type="button"
              onClick={() => navigate(`/runs/${resultRunId}/play`)}
              className="rounded-md bg-emerald-600 text-white px-3 py-1.5 text-sm hover:bg-emerald-700"
            >
              Try in playground →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
