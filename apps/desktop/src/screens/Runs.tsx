import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
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
  HubPushState,
  Run,
  ScheduledEntry,
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
  const navigate = useNavigate();
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Set of run ids the user has ticked for comparison. Limited to
  // SUCCEEDED runs since the chart only makes sense over completed
  // training. Not persisted — a fresh visit starts with no selection.
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!api) return;
    api.listRuns()
      .then((r) => {
        setError(null);
        setRuns(r.runs);
      })
      .catch((e: unknown) => setError(String(e)));
  }, [api]);

  function toggle(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  if (error) {
    return (
      <div className="p-6 space-y-2">
        <h1 className="text-2xl font-semibold">Couldn't reach the sidecar</h1>
        <pre className="text-xs bg-red-50 border border-red-200 rounded p-3 whitespace-pre-wrap">{error}</pre>
      </div>
    );
  }
  if (!runs) return <div className="p-6 text-zinc-500">Loading runs…</div>;

  const comparable = runs.filter((r) => r.status === "succeeded");
  const canCompare = selectedIds.size >= 2;
  const hasRuns = runs.length > 0;

  return (
    <div className="p-6 space-y-4">
      <header className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">Runs</h1>
        <div className="flex items-center gap-3">
          {selectedIds.size > 0 && (
            <span className="text-xs text-zinc-500">
              {selectedIds.size} selected
            </span>
          )}
          <button
            type="button"
            disabled={!canCompare}
            onClick={() =>
              navigate(`/runs/compare?ids=${Array.from(selectedIds).join(",")}`)
            }
            title={
              canCompare
                ? "Overlay loss curves from the selected runs."
                : "Tick at least two SUCCEEDED runs to compare."
            }
            className="text-xs px-3 py-1.5 rounded-md border border-zinc-300 text-zinc-700 hover:bg-zinc-50 disabled:opacity-50"
          >
            Compare ({selectedIds.size})
          </button>
          {selectedIds.size > 0 && (
            <button
              type="button"
              onClick={() => setSelectedIds(new Set())}
              className="text-xs text-zinc-500 hover:text-zinc-700"
            >
              clear
            </button>
          )}
        </div>
      </header>
      {comparable.length < 2 && (
        <p className="text-xs text-zinc-500">
          Compare needs at least two SUCCEEDED runs. Pending / failed /
          canceled runs aren't selectable.
        </p>
      )}
      <ScheduledSection />

      {!hasRuns && (
        <p className="text-zinc-500">
          No runs yet — start one from the Train page.
        </p>
      )}

      {hasRuns && (
      <ul className="divide-y divide-zinc-200 border border-zinc-200 rounded-lg">
        {runs.map((r) => {
          const selectable = r.status === "succeeded";
          const checked = selectedIds.has(r.id);
          return (
            <li key={r.id} className="flex items-center">
              <label
                className={`p-3 ${
                  selectable ? "cursor-pointer" : "cursor-not-allowed opacity-40"
                }`}
                title={
                  selectable
                    ? "Select for compare"
                    : `Cannot compare a ${r.status} run`
                }
              >
                <input
                  type="checkbox"
                  disabled={!selectable}
                  checked={checked}
                  onChange={() => toggle(r.id)}
                />
              </label>
              <Link
                to={`/runs/${r.id}`}
                className="flex-1 flex items-center justify-between p-3 hover:bg-zinc-50"
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
          );
        })}
      </ul>
      )}
    </div>
  );
}

export function RunDetail() {
  const api = useApiClient();
  const navigate = useNavigate();
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<Run | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [showResume, setShowResume] = useState(false);
  const [resumeEpochs, setResumeEpochs] = useState(1);
  const [resumeLr, setResumeLr] = useState<string>("");
  const [resuming, setResuming] = useState(false);
  const [resumeError, setResumeError] = useState<string | null>(null);
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
  const [latestLog, setLatestLog] = useState<string | null>(null);
  const [ggufQuant, setGgufQuant] = useState<GgufQuant>("q4_k_m");
  const [ggufExport, setGgufExport] = useState<GgufExportState | null>(null);
  const [ggufError, setGgufError] = useState<string | null>(null);
  const [hfSignedIn, setHfSignedIn] = useState<boolean | null>(null);
  const [hubRepo, setHubRepo] = useState("");
  const [hubPrivate, setHubPrivate] = useState(true);
  const [hubState, setHubState] = useState<HubPushState | null>(null);
  const [hubError, setHubError] = useState<string | null>(null);

  // Highest plotted step. Persists across SSE reconnects so the live stream
  // doesn't re-plot rows the events.jsonl replay already handled.
  const seenStepRef = useRef(-1);
  // Per-step timestamps used to compute steps/sec + ETA. We keep a
  // bounded ring of the most recent samples so the EMA reflects the
  // current pace (machines that throttle mid-run shouldn't carry the
  // earliest-step throughput forever).
  const stepTimingsRef = useRef<{ step: number; t: number }[]>([]);
  const STEP_TIMING_WINDOW = 30;
  // Live stats — recomputed on each step event. Exposed as state so
  // the header re-renders every tick.
  const [pace, setPace] = useState<{
    stepsPerSec: number;
    etaSeconds: number | null;
  } | null>(null);

  // Reset all per-run UI state when the runId changes. Without this, the
  // chart points / logs / seenStepRef from the previous run leak into the
  // next view: navigating from a run that hit step 100 to a fresh run
  // (max step 50) would silently discard every event because the new
  // step is "older" than the persisted seenStepRef. RunDetail is a
  // single component re-rendered with new params, so the state has to be
  // explicitly cleared.
  useEffect(() => {
    setRun(null);
    setPoints([]);
    setLogs([]);
    setDownload(null);
    setLatestLog(null);
    setStreamState("connecting");
    setRevealError(null);
    setGgufExport(null);
    setGgufError(null);
    setHubState(null);
    setHubError(null);
    seenStepRef.current = -1;
    stepTimingsRef.current = [];
    setPace(null);
  }, [runId]);

  useEffect(() => {
    if (!api || !runId) return;
    api.getRun(runId).then(setRun);
  }, [api, runId]);
  // Max points we keep on the chart. Beyond this, append-then-slice so
  // memory and React reconciliation cost stay bounded for runs that go
  // tens of thousands of steps. The chart already gets unreadable past a
  // few hundred points; trimming the tail just means losing the very
  // earliest steps, which matter least for "is loss still going down?".
  const MAX_CHART_POINTS = 2_000;

  function updatePace(currentStep: number, totalSteps: number) {
    const now = performance.now();
    const ring = stepTimingsRef.current;
    ring.push({ step: currentStep, t: now });
    while (ring.length > STEP_TIMING_WINDOW) ring.shift();
    if (ring.length < 2) {
      // Need at least two samples to compute a rate.
      return;
    }
    const first = ring[0];
    const last = ring[ring.length - 1];
    const dtSec = (last.t - first.t) / 1000;
    const dSteps = last.step - first.step;
    if (dtSec <= 0 || dSteps <= 0) return;
    const stepsPerSec = dSteps / dtSec;
    const remaining = totalSteps > 0 ? totalSteps - currentStep : 0;
    const etaSeconds = remaining > 0 ? remaining / stepsPerSec : null;
    setPace({ stepsPerSec, etaSeconds });
  }

  function formatEta(seconds: number): string {
    if (!Number.isFinite(seconds) || seconds < 0) return "—";
    if (seconds < 60) return `${Math.round(seconds)}s remaining`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    if (m < 60) return `${m}m ${s}s remaining`;
    const h = Math.floor(m / 60);
    const mm = m % 60;
    return `${h}h ${mm}m remaining`;
  }
  function applyEvent(type: string, p: TrainingEventPayload, opts: { live: boolean }) {
    if (type === "step" && p.loss !== null) {
      if (p.step > seenStepRef.current) {
        seenStepRef.current = p.step;
        setPoints((prev) => {
          const next = [...prev, { step: p.step, loss: p.loss as number }];
          return next.length > MAX_CHART_POINTS ? next.slice(-MAX_CHART_POINTS) : next;
        });
        // Only feed the pace calculator with live events. Replay
        // events are written to disk in real time but consumed in a
        // burst on mount, so their timestamps don't reflect the
        // training pace — including them would report a misleading
        // ~thousands-of-steps-per-second on the first frame.
        if (opts.live) {
          updatePace(p.step, p.total_steps);
        }
      }
      setDownload(null);
    }
    if (type === "download" && p.bytes_done !== null && p.bytes_total !== null) {
      // bytes_total can momentarily be 0 for files HF doesn't pre-announce
      // a size for. Skip those frames so the progress bar's width math
      // doesn't divide by zero and render NaN%.
      if (p.bytes_total > 0) {
        setDownload({
          bytesDone: p.bytes_done,
          bytesTotal: p.bytes_total,
          desc: p.message ?? "",
        });
      }
    }
    if (type === "log" && p.message) {
      setLatestLog(p.message);
    }
    const tag = `[${type}]`;
    const detail =
      type === "step"
        ? `step=${p.step}/${p.total_steps} loss=${p.loss?.toFixed(4) ?? "-"} lr=${p.lr ?? "-"}`
        : type === "download"
          ? `${p.message ?? "downloading"} ${p.bytes_done}/${p.bytes_total} bytes`
          : p.message ?? "";
    setLogs((prev) => [...prev, `${tag} ${detail}`].slice(-500));
    if (opts.live && (type === "done" || type === "error" || type === "canceled")) {
      setDownload(null);
      if (api && runId) api.getRun(runId).then(setRun);
    }
  }

  useEffect(() => {
    if (!api || !runId) return;
    let cancelled = false;
    api.getRunEvents(runId).then(({ events }) => {
      if (cancelled) return;
      for (const ev of events) {
        applyEvent(ev.type, ev, { live: false });
      }
    });
    return () => {
      cancelled = true;
    };
  }, [api, runId]);

  useEffect(() => {
    if (!api || !runId) return;
    const close = api.streamRun(
      runId,
      ({ type, payload }) => {
        applyEvent(type, payload as TrainingEventPayload, { live: true });
      },
      (state) => {
        setStreamState(state);
        if (state === "open") {
          api.getRun(runId).then(setRun);
        }
      },
    );
    return close;
  }, [api, runId]);

  const [cancelError, setCancelError] = useState<string | null>(null);

  async function handleResume() {
    if (!api || !runId) return;
    setResumeError(null);
    setResuming(true);
    try {
      const lrParsed = resumeLr.trim() === "" ? undefined : parseFloat(resumeLr);
      if (lrParsed !== undefined && !Number.isFinite(lrParsed)) {
        throw new Error("Learning rate isn't a valid number.");
      }
      const { id } = await api.resumeRun(runId, {
        epochs: resumeEpochs,
        learning_rate: lrParsed,
      });
      setShowResume(false);
      navigate(`/runs/${id}`);
    } catch (e) {
      setResumeError(String((e as Error).message ?? e));
    } finally {
      setResuming(false);
    }
  }

  async function handleDelete() {
    if (!api || !runId || !run) return;
    if (!window.confirm(
      "Delete this run? The adapter, logs, and any GGUF/merged files will be removed from disk. This can't be undone.",
    )) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await api.deleteRun(runId);
      navigate("/runs");
    } catch (e) {
      setDeleteError(String((e as Error).message ?? e));
      setDeleting(false);
    }
  }

  async function handleCancel() {
    if (!api || !runId || !run) return;
    setCanceling(true);
    setCancelError(null);
    try {
      await api.cancelRun(runId);
      // The trainer takes a moment to honor the signal; the SSE stream will
      // emit a "canceled" event when it does. Refresh state right away in case
      // the run was already finished.
      const fresh = await api.getRun(runId);
      setRun(fresh);
    } catch (e) {
      setCancelError(String((e as Error).message ?? e));
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

  // Re-check on mount and again after a push attempt — token may have been
  // added between visits via `huggingface-cli login` in a terminal.
  useEffect(() => {
    if (!api) return;
    api.getHfAuth().then((s) => setHfSignedIn(s.signed_in)).catch(() => undefined);
  }, [api]);

  // Resume any in-flight or completed hub push when navigating back to a
  // run. The push is async on the sidecar — without this, the user
  // wouldn't see "still pushing" or the completed URL on remount.
  useEffect(() => {
    if (!api || !runId) return;
    api.getHubExport(runId).then(setHubState).catch(() => undefined);
  }, [api, runId]);

  // Poll while a push is running. State file is small; 1.5s matches the
  // GGUF polling cadence and is fast enough that the upload's last_log
  // updates feel live.
  useEffect(() => {
    if (!api || !runId) return;
    if (hubState?.status !== "running") return;
    const id = setInterval(() => {
      api.getHubExport(runId).then((s) => {
        if (s) setHubState(s);
      });
    }, 1500);
    return () => clearInterval(id);
  }, [api, runId, hubState?.status]);

  async function handlePushToHub() {
    if (!api || !runId || !hubRepo.trim()) return;
    setHubError(null);
    try {
      const initial = await api.pushRunToHub(runId, {
        repo_id: hubRepo.trim(),
        private: hubPrivate,
      });
      setHubState(initial);
    } catch (e) {
      const err = e as Error & { status?: number };
      if (err.status === 401) {
        setHfSignedIn(false);
      }
      setHubError(err.message ?? String(e));
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
              title={
                canceling
                  ? "Cancellation is signaled — the trainer winds down at the next step boundary, which can take a moment if a step is mid-run."
                  : "Stop training. Already-saved adapter checkpoints stay on disk."
              }
              className="text-xs px-3 py-1 rounded-md border border-red-200 text-red-700 hover:bg-red-50 disabled:opacity-50"
            >
              {canceling ? "Canceling…" : "Cancel"}
            </button>
          )}
          {canceling && (
            <span className="text-xs text-zinc-500 italic">
              winding down at next step
            </span>
          )}
          {run.status === "succeeded" && (
            <Link
              to={`/runs/${run.id}/play`}
              title="Try the trained adapter on a prompt."
              className="text-xs px-3 py-1 rounded-md border border-blue-200 text-blue-700 hover:bg-blue-50"
            >
              Playground
            </Link>
          )}
          {run.status === "succeeded" && (
            <Link
              to={`/runs/${run.id}/eval`}
              title="Compare base vs adapter outputs side-by-side on a small prompt set."
              className="text-xs px-3 py-1 rounded-md border border-blue-200 text-blue-700 hover:bg-blue-50"
            >
              Eval
            </Link>
          )}
          {run.status === "succeeded" && (
            <button
              type="button"
              onClick={() => {
                setShowResume(true);
                setResumeError(null);
                setResumeEpochs(1);
                setResumeLr("");
              }}
              title="Continue training from this run's adapter."
              className="text-xs px-3 py-1 rounded-md border border-blue-200 text-blue-700 hover:bg-blue-50"
            >
              Continue training
            </button>
          )}
          {!isActive && (
            <button
              type="button"
              onClick={handleDelete}
              disabled={deleting}
              title="Remove this run from disk."
              className="text-xs px-3 py-1 rounded-md border border-zinc-300 text-zinc-700 hover:bg-zinc-50 disabled:opacity-50"
            >
              {deleting ? "Deleting…" : "Delete"}
            </button>
          )}
          <span className={`text-xs px-2 py-0.5 rounded-full ${STATUS_COLOR[run.status]}`}>
            {run.status}
          </span>
        </div>
      </header>

      {showResume && (
        <section className="rounded-lg border border-blue-200 bg-blue-50/40 p-4 space-y-3">
          <header className="flex items-baseline justify-between">
            <h2 className="text-sm font-medium">Continue training</h2>
            <button
              type="button"
              onClick={() => setShowResume(false)}
              className="text-xs text-zinc-500 hover:text-zinc-700"
            >
              cancel
            </button>
          </header>
          <p className="text-xs text-zinc-600 leading-relaxed">
            Spawns a new run that picks up from this run's adapter weights.
            The base model, dataset, LoRA rank/alpha stay the same — only
            the epoch count (and optionally a smaller learning rate) are
            different. The current run is preserved so you can branch or
            roll back.
          </p>
          <div className="grid grid-cols-2 gap-3">
            <label className="text-sm space-y-1 block">
              <span className="block text-xs text-zinc-600">
                Additional epochs
              </span>
              <input
                type="number"
                min={1}
                value={resumeEpochs}
                onChange={(e) => {
                  const next = parseInt(e.target.value, 10);
                  setResumeEpochs(Number.isFinite(next) && next > 0 ? next : resumeEpochs);
                }}
                className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
              />
            </label>
            <label className="text-sm space-y-1 block">
              <span className="block text-xs text-zinc-600">
                Learning rate (blank = inherit)
              </span>
              <input
                type="text"
                inputMode="decimal"
                value={resumeLr}
                placeholder={String(run.config.learning_rate ?? "2e-4")}
                onChange={(e) => setResumeLr(e.target.value)}
                className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm font-mono"
              />
            </label>
          </div>
          {resumeError && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
              {resumeError}
            </div>
          )}
          <button
            type="button"
            onClick={handleResume}
            disabled={resuming || resumeEpochs < 1}
            className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
          >
            {resuming ? "Starting…" : "Start continuation"}
          </button>
        </section>
      )}

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

      {isActive && pace && (
        <div className="flex items-center gap-3 text-xs text-zinc-600">
          <span className="font-mono tabular-nums">
            {pace.stepsPerSec >= 1
              ? `${pace.stepsPerSec.toFixed(1)} steps/s`
              : `${(1 / pace.stepsPerSec).toFixed(1)}s/step`}
          </span>
          {pace.etaSeconds !== null && (
            <span className="font-mono tabular-nums text-zinc-500">
              · {formatEta(pace.etaSeconds)}
            </span>
          )}
        </div>
      )}

      <section className="h-64 rounded-lg border border-zinc-200 p-4">
        {points.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center gap-1 text-sm text-zinc-400 px-4 text-center">
            <div>
              {download
                ? "Waiting for download to finish…"
                : "Waiting for first step…"}
            </div>
            {latestLog && (
              <div className="text-xs font-mono text-zinc-500 truncate max-w-full">
                {latestLog}
              </div>
            )}
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
          <p className="text-xs text-zinc-600 leading-relaxed">
            Merges the LoRA adapter into the base weights and converts the
            result to a single .gguf file you can load in llama.cpp, Ollama,
            or LM Studio. <span className="font-medium">q4_k_m</span> is the
            best size/quality tradeoff for most users; pick a higher quant if
            you have RAM to spare.
          </p>
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
            <div className="space-y-1">
              <div className="text-sm text-zinc-700 flex items-center gap-2">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                {ggufExport.step === "convert"
                  ? `Converting to ${ggufExport.quant}…`
                  : "Merging adapter into base weights…"}
              </div>
              {ggufExport.latest_log && (
                <div className="text-xs font-mono text-zinc-500 truncate">
                  {ggufExport.latest_log}
                </div>
              )}
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
              <div className="whitespace-pre-wrap">Export failed: {ggufExport.error}</div>
              {ggufExport.merged_path && (
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      await revealItemInDir(ggufExport.merged_path as string);
                    } catch (e) {
                      setRevealError(String(e));
                    }
                  }}
                  className="text-xs px-3 py-1 rounded-md border border-zinc-300 text-zinc-700 hover:bg-zinc-50"
                >
                  Reveal merged model
                </button>
              )}
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

      {run.status === "succeeded" && (
        <section className="rounded-lg border border-zinc-200 p-4 space-y-3">
          <div className="flex items-baseline justify-between">
            <h2 className="text-sm font-medium">Push to Hugging Face Hub</h2>
            {hfSignedIn === false && (
              <span className="text-xs text-amber-700">Not signed in</span>
            )}
          </div>
          <p className="text-xs text-zinc-600 leading-relaxed">
            Uploads the trained LoRA adapter to a repo on Hugging Face. Anyone
            with access to the repo can then load it on top of the base model.
            Auth comes from <code className="font-mono">huggingface-cli login</code>;
            we never store your token.
          </p>
          {hfSignedIn === false && (
            <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2">
              No HF token detected. Run{" "}
              <code className="font-mono">huggingface-cli login</code> in a
              terminal, then come back to this screen.
            </div>
          )}
          {(!hubState || hubState.status === "failed") && (
            <div className="space-y-2">
              <label className="text-sm space-y-1 block">
                <span className="block text-xs text-zinc-600">Repo ID</span>
                <input
                  value={hubRepo}
                  onChange={(e) => setHubRepo(e.target.value)}
                  placeholder="username/my-adapter"
                  className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm font-mono"
                />
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={hubPrivate}
                  onChange={(e) => setHubPrivate(e.target.checked)}
                />
                <span>Create as private repo</span>
              </label>
              <button
                type="button"
                onClick={handlePushToHub}
                disabled={!hubRepo.trim() || hfSignedIn === false}
                className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
              >
                Push adapter
              </button>
            </div>
          )}
          {hubState?.status === "running" && (
            <div className="space-y-1">
              <div className="text-sm text-zinc-700 flex items-center gap-2">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                Uploading to {hubState.repo_id}…
              </div>
              {hubState.latest_log && (
                <div className="text-xs font-mono text-zinc-500 truncate">
                  {hubState.latest_log}
                </div>
              )}
            </div>
          )}
          {hubState?.status === "done" && hubState.url && (
            <div className="space-y-2">
              <div className="text-sm text-green-700">
                Pushed to{" "}
                <a
                  href={hubState.url}
                  target="_blank"
                  rel="noreferrer"
                  className="font-mono underline"
                >
                  {hubState.url}
                </a>
              </div>
              <button
                type="button"
                onClick={() => {
                  setHubState(null);
                  setHubRepo("");
                }}
                className="text-xs px-3 py-1 rounded-md border border-zinc-300 text-zinc-700 hover:bg-zinc-50"
              >
                Push again
              </button>
            </div>
          )}
          {hubState?.status === "failed" && hubState.error && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2 whitespace-pre-wrap">
              Push failed: {hubState.error}
            </div>
          )}
          {hubError && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
              {hubError}
            </div>
          )}
        </section>
      )}

      {revealError && (
        <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
          Couldn't open the output directory: {revealError}
        </div>
      )}

      {cancelError && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
          Cancel failed: {cancelError}
        </div>
      )}

      {deleteError && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
          Delete failed: {deleteError}
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


function ScheduledSection() {
  const api = useApiClient();
  const [scheduled, setScheduled] = useState<ScheduledEntry[]>([]);
  const [busy, setBusy] = useState<string | null>(null);

  function refresh() {
    if (!api) return;
    api.listScheduledRuns().then((r) => setScheduled(r.scheduled));
  }

  useEffect(() => {
    refresh();
    // Lightweight poll so the section reflects entries firing while
    // the user has the page open. 5s is long enough to not hammer the
    // sidecar but short enough that "I just canceled" updates feel
    // immediate without manual refresh.
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  if (scheduled.length === 0) return null;

  async function cancel(id: string) {
    if (!api) return;
    setBusy(id);
    try {
      await api.cancelScheduledRun(id);
    } finally {
      setBusy(null);
      refresh();
    }
  }

  return (
    <section className="space-y-2 border-t border-zinc-200 pt-4">
      <h2 className="text-sm font-semibold text-zinc-700">
        Scheduled · {scheduled.length}
      </h2>
      <ul className="space-y-1 text-sm">
        {scheduled.map((s) => {
          const startsAt = new Date(s.start_at);
          const isPast = startsAt.getTime() < Date.now();
          return (
            <li
              key={s.id}
              className="flex items-center gap-3 px-3 py-2 rounded-md border border-zinc-200 bg-white"
            >
              <span
                className={`text-[10px] px-2 py-0.5 rounded uppercase tracking-wide ${
                  isPast
                    ? "bg-amber-100 text-amber-800"
                    : "bg-blue-100 text-blue-800"
                }`}
              >
                {isPast ? "missed" : "pending"}
              </span>
              <span className="text-zinc-700 flex-1 truncate">
                {startsAt.toLocaleString()} — {s.config.model_id}
              </span>
              <span className="text-xs text-zinc-500 font-mono">
                {s.id.slice(0, 8)}
              </span>
              <button
                type="button"
                onClick={() => cancel(s.id)}
                disabled={busy === s.id}
                className="text-xs px-2 py-1 rounded border border-zinc-300 hover:bg-zinc-50 disabled:opacity-50"
              >
                {busy === s.id ? "Canceling…" : "Cancel"}
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
