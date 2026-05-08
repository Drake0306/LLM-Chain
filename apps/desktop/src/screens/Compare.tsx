import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Run, TrainingEventPayload } from "../api/client";
import { useApiClient } from "../api/hooks";

/**
 * Loss curves for two or more SUCCEEDED runs overlaid on a single chart.
 * Lets the user eyeball whether a hyperparameter sweep helped — without
 * hand-comparing two RunDetail tabs.
 *
 * Wire: query string `?ids=a,b,c`. The list page sets it; refreshing
 * the page picks the same set up. Each id is fetched in parallel for
 * its run config + replayed events, then merged into one wide chart
 * dataset where each row carries the loss for every selected run at
 * that step (or null if that run was already done by then).
 */

/**
 * Recommendation banner shown when Compare is in "LR finder" mode
 * (?live=1, set by the Train page when launching the sweep).
 *
 * Picks the run whose final loss is lowest among the SUCCEEDED ones.
 * NaN / non-finite final losses are excluded — they signal a
 * runaway-LR run rather than a candidate to recommend. While runs
 * are still in flight we render a "still running" line; once all
 * have reached a terminal state we render the recommendation.
 */
function LrFinderBanner({ series }: { series: RunSeries[] | null }) {
  if (!series) return null;
  const inFlight = series.filter(
    (s) => s.run.status === "pending" || s.run.status === "running",
  );
  if (inFlight.length > 0) {
    return (
      <div className="text-xs text-zinc-600 bg-zinc-50 border border-zinc-200 rounded p-2 flex items-center gap-2">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
        Sniff in progress: {series.length - inFlight.length} of{" "}
        {series.length} runs done. Banner will recommend an LR once
        all runs finish.
      </div>
    );
  }
  // All terminal. Find the lowest final loss among SUCCEEDED runs.
  const candidates = series
    .filter((s) => s.run.status === "succeeded")
    .map((s) => {
      const losses = [...s.byStep.values()].filter((v) => Number.isFinite(v));
      const finalLoss = losses[losses.length - 1];
      return { run: s.run, finalLoss };
    })
    .filter((c) => c.finalLoss !== undefined);
  if (candidates.length === 0) {
    return (
      <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2">
        No SUCCEEDED runs in this sweep — every learning rate failed
        before completing. Try a smaller batch size or check the
        dataset before re-running.
      </div>
    );
  }
  const winner = candidates.reduce((best, c) =>
    (c.finalLoss as number) < (best.finalLoss as number) ? c : best,
  );
  const lr = winner.run.config.learning_rate;
  const stepBudget = winner.run.config.max_steps;
  return (
    <div className="text-sm text-green-800 bg-green-50 border border-green-200 rounded p-3 leading-relaxed">
      Recommended learning rate:{" "}
      <span className="font-mono font-medium">{lr}</span>
      {" "}— lowest final loss{" "}
      <span className="font-mono">{(winner.finalLoss as number).toFixed(4)}</span>
      {stepBudget ? ` at ${stepBudget} steps` : ""}. Use this in the Train
      page LR field for the full run.
    </div>
  );
}

// Stable palette so two-run comparisons get the same colors as larger
// ones. Ten entries covers any realistic hand-curated compare set,
// and we cap the URL-driven selection at this length so a paste of
// 1000 ids can't fan out to 1000 parallel API calls.
const COLORS = [
  "#2563eb", // blue
  "#dc2626", // red
  "#16a34a", // green
  "#d97706", // amber
  "#7c3aed", // violet
  "#0891b2", // cyan
  "#db2777", // pink
  "#65a30d", // lime
  "#9333ea", // purple
  "#0d9488", // teal
];
const MAX_RUNS = COLORS.length;

interface RunSeries {
  run: Run;
  /** Sparse map of step → loss. Some runs may have skipped logging
   * steps; we just plot what we received. */
  byStep: Map<number, number>;
}

export function Compare() {
  const api = useApiClient();
  const [params] = useSearchParams();
  const { ids, truncated, live } = useMemo(() => {
    const all = (params.get("ids") ?? "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    return {
      ids: all.slice(0, MAX_RUNS),
      truncated: all.length > MAX_RUNS,
      // ?live=1 enables auto-polling — set by the LR finder so the
      // compare view fills in as the sequential mini-runs finish.
      // Plain Compare links don't poll (the runs are already done).
      live: params.get("live") === "1",
    };
  }, [params]);
  const [series, setSeries] = useState<RunSeries[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!api || ids.length === 0) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    setSeries(null);
    setError(null);

    async function fetchOnce(): Promise<RunSeries[]> {
      return Promise.all(
        ids.map(async (id) => {
          const [run, eventsResp] = await Promise.all([
            api!.getRun(id),
            api!.getRunEvents(id),
          ]);
          const byStep = new Map<number, number>();
          for (const ev of eventsResp.events as TrainingEventPayload[]) {
            if (ev.type === "step" && typeof ev.loss === "number") {
              byStep.set(ev.step, ev.loss);
            }
          }
          return { run, byStep } satisfies RunSeries;
        }),
      );
    }

    function tick() {
      fetchOnce()
        .then((all) => {
          if (cancelled) return;
          setSeries(all);
          // Keep polling while at least one run isn't terminal —
          // the LR finder writes events progressively as each child
          // runs. Stop once everyone's settled to save battery.
          if (live && all.some((s) => s.run.status === "running" || s.run.status === "pending")) {
            timer = setTimeout(tick, 1500);
          }
        })
        .catch((e: unknown) => {
          if (!cancelled) setError(String((e as Error).message ?? e));
        });
    }

    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [api, ids, live]);

  // Wide-format chart data. Recharts works best with one row per X value
  // and each series as a column; we walk the union of step indices and
  // emit one entry per step.
  const chartData = useMemo(() => {
    if (!series) return [];
    const allSteps = new Set<number>();
    for (const s of series) for (const k of s.byStep.keys()) allSteps.add(k);
    const sorted = [...allSteps].sort((a, b) => a - b);
    return sorted.map((step) => {
      const row: Record<string, number | null> = { step };
      for (const s of series) {
        const v = s.byStep.get(step);
        row[s.run.id] = v === undefined ? null : v;
      }
      return row;
    });
  }, [series]);

  if (ids.length < 2) {
    return (
      <div className="p-6 space-y-3">
        <h1 className="text-2xl font-semibold">Compare runs</h1>
        <p className="text-sm text-zinc-600">
          Pick at least two runs on the{" "}
          <Link to="/runs" className="underline">
            Runs
          </Link>{" "}
          page and click Compare.
        </p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="p-6 space-y-2">
        <h1 className="text-2xl font-semibold">Compare runs</h1>
        <pre className="text-xs bg-red-50 border border-red-200 rounded p-3 whitespace-pre-wrap">
          {error}
        </pre>
      </div>
    );
  }
  if (!series) {
    return <div className="p-6 text-zinc-500">Loading runs…</div>;
  }

  return (
    <div className="p-6 space-y-6">
      <header className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">Compare runs</h1>
        <Link to="/runs" className="text-sm text-zinc-600 hover:underline">
          ← back to Runs
        </Link>
      </header>
      {truncated && (
        <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2">
          Comparing the first {MAX_RUNS} runs only — the chart's color
          palette tops out there. Trim your selection if you need other
          runs.
        </p>
      )}
      {live && <LrFinderBanner series={series} />}

      <section className="h-80 rounded-lg border border-zinc-200 p-4">
        {chartData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-sm text-zinc-400">
            No step events recorded for any selected run.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} />
              <YAxis />
              <Tooltip
                formatter={(value) =>
                  typeof value === "number" ? value.toFixed(4) : "—"
                }
              />
              <Legend />
              {series.map((s, i) => (
                <Line
                  key={s.run.id}
                  type="monotone"
                  dataKey={s.run.id}
                  stroke={COLORS[i % COLORS.length]}
                  dot={false}
                  isAnimationActive={false}
                  // Show short id in legend; full id is in the side panel.
                  name={s.run.id}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {series.map((s, i) => {
          const cfg = s.run.config;
          const losses = [...s.byStep.values()];
          const finalLoss = losses[losses.length - 1];
          return (
            <article
              key={s.run.id}
              className="rounded-lg border border-zinc-200 p-4 space-y-2"
            >
              <header className="flex items-baseline justify-between gap-3">
                <Link
                  to={`/runs/${s.run.id}`}
                  className="font-mono text-xs text-zinc-700 hover:underline"
                >
                  {s.run.id}
                </Link>
                <span
                  className="inline-block w-3 h-3 rounded-full"
                  style={{ backgroundColor: COLORS[i % COLORS.length] }}
                  aria-label="series color"
                />
              </header>
              <div className="text-sm font-medium">{cfg.model_id}</div>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-zinc-600">
                <dt className="text-zinc-500">Backend</dt>
                <dd className="font-mono">{cfg.backend}</dd>
                <dt className="text-zinc-500">Technique</dt>
                <dd className="font-mono">{cfg.technique}</dd>
                <dt className="text-zinc-500">Epochs</dt>
                <dd className="font-mono">{cfg.epochs ?? 1}</dd>
                <dt className="text-zinc-500">Batch</dt>
                <dd className="font-mono">{cfg.batch_size ?? 1}</dd>
                <dt className="text-zinc-500">LR</dt>
                <dd className="font-mono">{cfg.learning_rate ?? "—"}</dd>
                <dt className="text-zinc-500">LoRA r/α</dt>
                <dd className="font-mono">
                  {cfg.lora_rank ?? "—"} / {cfg.lora_alpha ?? "—"}
                </dd>
                <dt className="text-zinc-500">Steps</dt>
                <dd className="font-mono">{losses.length}</dd>
                <dt className="text-zinc-500">Final loss</dt>
                <dd className="font-mono">
                  {finalLoss === undefined ? "—" : finalLoss.toFixed(4)}
                </dd>
              </dl>
            </article>
          );
        })}
      </section>
    </div>
  );
}
