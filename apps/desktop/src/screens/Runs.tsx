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

import type {
  GgufExportState,
  GgufQuant,
  Run,
  StreamState,
  TrainingEventPayload,
} from "../api/client";
import { useApiClient } from "../api/hooks";

const GGUF_QUANTS: { value: GgufQuant; label: string; help: string }[] = [
  { value: "q4_k_m", label: "q4_k_m (4-bit, recommended)", help: "Smallest, runs on most laptops; needs llama-quantize built." },
  { value: "q8_0", label: "q8_0 (8-bit)", help: "Higher fidelity; works without llama-quantize." },
  { value: "f16", label: "f16 (no quantization)", help: "Largest file; works without llama-quantize." },
];

function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

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
  const [download, setDownload] = useState<{
    bytesDone: number;
    bytesTotal: number;
    desc: string;
  } | null>(null);
  const [ggufQuant, setGgufQuant] = useState<GgufQuant>("q4_k_m");
  const [ggufExport, setGgufExport] = useState<GgufExportState | null>(null);
  const [ggufError, setGgufError] = useState<string | null>(null);
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
          // First training step means downloads are done — clear the bar.
          setDownload(null);
        }
        if (type === "download" && p.bytes_done !== null && p.bytes_total !== null) {
          setDownload({
            bytesDone: p.bytes_done,
            bytesTotal: p.bytes_total,
            desc: p.message ?? "",
          });
        }
        const tag = `[${type}]`;
        const detail =
          type === "step"
            ? `step=${p.step}/${p.total_steps} loss=${p.loss?.toFixed(4) ?? "-"} lr=${p.lr ?? "-"}`
            : type === "download"
              ? `${p.message ?? "downloading"} ${p.bytes_done}/${p.bytes_total} bytes`
              : p.message ?? "";
        setLogs((prev) => [...prev, `${tag} ${detail}`].slice(-500));
        if (type === "done" || type === "error" || type === "canceled") {
          setDownload(null);
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

  // Resume the export panel state when navigating back to a run that's
  // already been exported once. Without this the user would see the "Export"
  // button again instead of the .gguf path they already produced.
  useEffect(() => {
    if (!api || !runId) return;
    api.getGgufExport(runId).then(setGgufExport).catch(() => undefined);
  }, [api, runId]);

  // Poll while an export is running. The sidecar runs the merge+convert in
  // a background thread and writes status to disk; the UI just reads it back.
  useEffect(() => {
    if (!api || !runId) return;
    if (ggufExport?.status !== "running") return;
    const id = setInterval(() => {
      api.getGgufExport(runId).then((s) => {
        if (s) setGgufExport(s);
      });
    }, 1500);
    return () => clearInterval(id);
  }, [api, runId, ggufExport?.status]);

  async function handleStartExport() {
    if (!api || !runId) return;
    setGgufError(null);
    try {
      const state = await api.startGgufExport(runId, ggufQuant);
      setGgufExport(state);
    } catch (e) {
      setGgufError(String((e as Error).message ?? e));
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

      {download && (
        <section className="rounded-lg border border-zinc-200 p-4 space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-zinc-700">
              Downloading {download.desc || run.config.model_id}
            </span>
            <span className="font-mono text-zinc-500">
              {formatBytes(download.bytesDone)} / {formatBytes(download.bytesTotal)}
            </span>
          </div>
          <div className="h-1.5 bg-zinc-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-[width] duration-150"
              style={{
                width: `${Math.min(
                  100,
                  Math.round((download.bytesDone / download.bytesTotal) * 100),
                )}%`,
              }}
            />
          </div>
        </section>
      )}

      <section className="h-64 rounded-lg border border-zinc-200 p-4">
        {points.length === 0 ? (
          <div className="h-full flex items-center justify-center text-sm text-zinc-400">
            {download ? "Waiting for download to finish…" : "Waiting for first step…"}
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

      {run.status === "succeeded" && (
        <section className="rounded-lg border border-zinc-200 p-4 space-y-3">
          <div className="flex items-baseline justify-between">
            <h2 className="text-sm font-medium">Export to GGUF</h2>
            <span className="text-xs text-zinc-500">
              For llama.cpp / Ollama / LM Studio
            </span>
          </div>
          {!ggufExport && (
            <div className="flex items-end gap-3">
              <label className="text-sm space-y-1 flex-1">
                <span className="block text-xs text-zinc-600">Quantization</span>
                <select
                  value={ggufQuant}
                  onChange={(e) => setGgufQuant(e.target.value as GgufQuant)}
                  className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
                >
                  {GGUF_QUANTS.map((q) => (
                    <option key={q.value} value={q.value}>
                      {q.label}
                    </option>
                  ))}
                </select>
                <span className="block text-xs text-zinc-500">
                  {GGUF_QUANTS.find((q) => q.value === ggufQuant)?.help}
                </span>
              </label>
              <button
                type="button"
                onClick={handleStartExport}
                className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm font-medium"
              >
                Export
              </button>
            </div>
          )}
          {ggufExport?.status === "running" && (
            <div className="text-sm text-zinc-700 flex items-center gap-2">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
              {ggufExport.step === "convert"
                ? `Converting to ${ggufExport.quant}…`
                : "Merging adapter into base weights…"}
            </div>
          )}
          {ggufExport?.status === "done" && ggufExport.path && (
            <div className="space-y-2">
              <div className="text-sm text-green-700">
                Exported to <span className="font-mono">{ggufExport.path}</span>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      await revealItemInDir(ggufExport.path as string);
                    } catch (e) {
                      setRevealError(String(e));
                    }
                  }}
                  className="text-xs px-3 py-1 rounded-md border border-zinc-300 text-zinc-700 hover:bg-zinc-50"
                >
                  Reveal GGUF
                </button>
                <button
                  type="button"
                  onClick={() => setGgufExport(null)}
                  className="text-xs px-3 py-1 rounded-md border border-zinc-300 text-zinc-700 hover:bg-zinc-50"
                >
                  Export another quant
                </button>
              </div>
            </div>
          )}
          {ggufExport?.status === "failed" && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2 space-y-2">
              <div>Export failed: {ggufExport.error}</div>
              <button
                type="button"
                onClick={() => setGgufExport(null)}
                className="text-xs px-3 py-1 rounded-md border border-red-200 text-red-700 hover:bg-red-100"
              >
                Try again
              </button>
            </div>
          )}
          {ggufError && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
              {ggufError}
            </div>
          )}
        </section>
      )}

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
