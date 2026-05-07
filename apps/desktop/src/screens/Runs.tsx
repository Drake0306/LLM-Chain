import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Run, StreamState, TrainingEventPayload } from "../api/client";
import { useApiClient } from "../api/hooks";

const STATUS_COLOR: Record<Run["status"], string> = {
  pending: "bg-zinc-100 text-zinc-700",
  running: "bg-blue-100 text-blue-800",
  succeeded: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
  canceled: "bg-zinc-200 text-zinc-700",
};

export function RunsList() {
  const api = useApiClient();
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!api) return;
    api.listRuns()
      .then((r) => {
        setError(null);
        setRuns(r.runs);
      })
      .catch((e: unknown) => setError(String(e)));
  }, [api]);

  if (error) {
    return (
      <div className="p-6 space-y-2">
        <h1 className="text-2xl font-semibold">Couldn't reach the sidecar</h1>
        <pre className="text-xs bg-red-50 border border-red-200 rounded p-3 whitespace-pre-wrap">{error}</pre>
      </div>
    );
  }
  if (!runs) return <div className="p-6 text-zinc-500">Loading runs…</div>;
  if (runs.length === 0)
    return <div className="p-6 text-zinc-500">No runs yet — start one from the Train page.</div>;

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-2xl font-semibold">Runs</h1>
      <ul className="divide-y divide-zinc-200 border border-zinc-200 rounded-lg">
        {runs.map((r) => (
          <li key={r.id}>
            <Link
              to={`/runs/${r.id}`}
              className="flex items-center justify-between p-3 hover:bg-zinc-50"
            >
              <div className="space-y-1">
                <div className="font-mono text-xs text-zinc-500">{r.id}</div>
                <div className="text-sm">
                  {r.config.model_id} · {r.config.technique.toUpperCase()} · {r.config.backend}
                </div>
              </div>
              <span className={`text-xs px-2 py-0.5 rounded-full ${STATUS_COLOR[r.status]}`}>
                {r.status}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function RunDetail() {
  const api = useApiClient();
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<Run | null>(null);
  const [points, setPoints] = useState<{ step: number; loss: number }[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [canceling, setCanceling] = useState(false);
  const [streamState, setStreamState] = useState<StreamState>("connecting");
  const [revealError, setRevealError] = useState<string | null>(null);
  const startedRef = useRef(false);

  useEffect(() => {
    if (!api || !runId) return;
    api.getRun(runId).then(setRun);
  }, [api, runId]);

  useEffect(() => {
    if (!api || !runId || startedRef.current) return;
    startedRef.current = true;
    const close = api.streamRun(
      runId,
      ({ type, payload }) => {
        const p = payload as TrainingEventPayload;
        if (type === "step" && p.loss !== null) {
          setPoints((prev) => [...prev, { step: p.step, loss: p.loss as number }]);
        }
        const tag = `[${type}]`;
        const detail =
          type === "step"
            ? `step=${p.step}/${p.total_steps} loss=${p.loss?.toFixed(4) ?? "-"} lr=${p.lr ?? "-"}`
            : p.message ?? "";
        setLogs((prev) => [...prev, `${tag} ${detail}`].slice(-500));
        if (type === "done" || type === "error" || type === "canceled") {
          api.getRun(runId).then(setRun);
        }
      },
      (state) => {
        setStreamState(state);
        // When EventSource transitions through reconnecting -> open we want to
        // re-sync the run state in case we missed a terminal event during the
        // gap. The browser handles the actual retry; we just observe.
        if (state === "open") {
          api.getRun(runId).then(setRun);
        }
      },
    );
    return close;
  }, [api, runId]);

  async function handleCancel() {
    if (!api || !runId || !run) return;
    setCanceling(true);
    try {
      await api.cancelRun(runId);
      // The trainer takes a moment to honor the signal; the SSE stream will
      // emit a "canceled" event when it does. Refresh state right away in case
      // the run was already finished.
      const fresh = await api.getRun(runId);
      setRun(fresh);
    } finally {
      setCanceling(false);
    }
  }

  if (!run) return <div className="p-6 text-zinc-500">Loading…</div>;

  const isActive = run.status === "running" || run.status === "pending";

  return (
    <div className="p-6 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">{run.config.model_id}</h1>
          <p className="text-sm text-zinc-500 font-mono">{run.id}</p>
        </div>
        <div className="flex items-center gap-3">
          {isActive && streamState === "reconnecting" && (
            <span className="flex items-center gap-1.5 text-xs text-amber-700">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
              Reconnecting…
            </span>
          )}
          {run.output_dir && (
            <button
              type="button"
              onClick={async () => {
                try {
                  setRevealError(null);
                  await revealItemInDir(run.output_dir as string);
                } catch (e) {
                  setRevealError(String(e));
                }
              }}
              className="text-xs px-3 py-1 rounded-md border border-zinc-300 text-zinc-700 hover:bg-zinc-50"
              title={run.output_dir}
            >
              Reveal in {navigator.platform.startsWith("Mac") ? "Finder" : "Explorer"}
            </button>
          )}
          {isActive && (
            <button
              type="button"
              onClick={handleCancel}
              disabled={canceling}
              className="text-xs px-3 py-1 rounded-md border border-red-200 text-red-700 hover:bg-red-50 disabled:opacity-50"
            >
              {canceling ? "Canceling…" : "Cancel"}
            </button>
          )}
          <span className={`text-xs px-2 py-0.5 rounded-full ${STATUS_COLOR[run.status]}`}>
            {run.status}
          </span>
        </div>
      </header>

      <section className="h-64 rounded-lg border border-zinc-200 p-4">
        {points.length === 0 ? (
          <div className="h-full flex items-center justify-center text-sm text-zinc-400">
            Waiting for first step…
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="loss" stroke="#2563eb" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </section>

      <section>
        <h2 className="text-sm font-medium mb-2">Log</h2>
        <pre className="bg-zinc-900 text-zinc-100 text-xs font-mono p-3 rounded-md max-h-64 overflow-auto whitespace-pre-wrap">
          {logs.join("\n") || "—"}
        </pre>
      </section>

      {revealError && (
        <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
          Couldn't open the output directory: {revealError}
        </div>
      )}

      {run.error && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-3">
          {run.error}
        </div>
      )}
    </div>
  );
}
