import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import type { CuratedDownloadState, CuratedEntry } from "../api/client";
import { useApiClient } from "../api/hooks";
import { useSelection } from "../state/selection";

export function DatasetCurated() {
  const api = useApiClient();
  const navigate = useNavigate();
  const { setDataset } = useSelection();

  const [entries, setEntries] = useState<CuratedEntry[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Per-entry download state. Mirrors the sidecar's polling contract:
  // null = idle (no request started); object = the latest snapshot
  // from the status endpoint. Only one entry's state is "running" at
  // a time in the typical UX, but we track per-id so a user mashing
  // two cards doesn't desync.
  const [states, setStates] = useState<Record<string, CuratedDownloadState>>(
    {},
  );
  const [polling, setPolling] = useState<Set<string>>(new Set());
  const pollingRef = useRef(polling);
  pollingRef.current = polling;

  useEffect(() => {
    if (!api) return;
    let cancelled = false;
    api
      .listCuratedDatasets()
      .then((r) => !cancelled && setEntries(r.datasets))
      .catch((e: unknown) =>
        !cancelled && setLoadError(String((e as Error).message ?? e)),
      );
    return () => {
      cancelled = true;
    };
  }, [api]);

  // Poll loop — runs every 1s while at least one entry is downloading.
  // Stops when all polled entries reach a terminal state. Cheaper than
  // per-entry intervals and keeps the cleanup story simple.
  useEffect(() => {
    if (!api || polling.size === 0) return;
    let cancelled = false;
    const tick = async () => {
      if (cancelled || !api) return;
      const ids = Array.from(pollingRef.current);
      const results = await Promise.all(
        ids.map((id) =>
          api
            .getCuratedDownloadStatus(id)
            .then((s) => [id, s] as const)
            .catch(() => [id, null] as const),
        ),
      );
      if (cancelled) return;
      setStates((prev) => {
        const next = { ...prev };
        for (const [id, s] of results) {
          if (s) next[id] = s;
        }
        return next;
      });
      setPolling((prev) => {
        const next = new Set(prev);
        for (const [id, s] of results) {
          if (s && (s.status === "done" || s.status === "failed")) {
            next.delete(id);
          }
        }
        return next;
      });
    };
    const handle = setInterval(tick, 1000);
    // Kick once immediately so the user sees movement without a 1s wait.
    tick();
    return () => {
      cancelled = true;
      clearInterval(handle);
    };
  }, [api, polling]);

  async function startDownload(entry: CuratedEntry) {
    if (!api) return;
    setStates((prev) => ({
      ...prev,
      [entry.id]: { status: "running", id: entry.id },
    }));
    setPolling((prev) => {
      const next = new Set(prev);
      next.add(entry.id);
      return next;
    });
    try {
      await api.startCuratedDownload(entry.id);
    } catch (e) {
      setStates((prev) => ({
        ...prev,
        [entry.id]: {
          status: "failed",
          error: String((e as Error).message ?? e),
        },
      }));
      setPolling((prev) => {
        const next = new Set(prev);
        next.delete(entry.id);
        return next;
      });
    }
  }

  function useInTrain(entry: CuratedEntry) {
    const s = states[entry.id];
    if (!s || !s.path) return;
    setDataset({ format: "jsonl_chat", path: s.path });
    navigate("/train");
  }

  return (
    <div className="p-6 max-w-5xl space-y-6">
      <header className="flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Curated datasets</h1>
          <p className="text-sm text-zinc-500 leading-relaxed max-w-3xl">
            Vetted fine-tune datasets the sidecar can download from
            Hugging Face on click. Each entry shows its license front-
            and-centre — review before downloading; especially before
            training a model you intend to distribute.
          </p>
        </div>
        <Link
          to="/dataset"
          className="text-sm text-zinc-600 hover:text-zinc-900"
        >
          ← Back to Dataset picker
        </Link>
      </header>

      {loadError && (
        <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded p-2 leading-relaxed">
          {loadError}
        </div>
      )}

      {!entries && !loadError && (
        <div className="text-sm text-zinc-500">Loading manifest…</div>
      )}

      <div className="space-y-3">
        {(entries ?? []).map((entry) => {
          const state = states[entry.id];
          const isRunning = state?.status === "running";
          const isDone = state?.status === "done" && state.path;
          return (
            <div
              key={entry.id}
              className="rounded-lg border border-zinc-200 bg-white p-4 space-y-2"
            >
              <div className="flex items-baseline justify-between gap-3">
                <div>
                  <h2 className="text-lg font-medium">{entry.name}</h2>
                  <div className="text-xs font-mono text-zinc-500">
                    {entry.hf_id}
                  </div>
                </div>
                <div className="text-right text-xs">
                  <div className="text-zinc-700">
                    {entry.size_rows.toLocaleString()} rows
                    {entry.size_mb > 0 && ` · ~${entry.size_mb} MB`}
                  </div>
                </div>
              </div>
              <p className="text-sm text-zinc-700 leading-relaxed">
                {entry.description}
              </p>
              <div className="flex items-center gap-3 text-xs">
                <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-900 font-medium">
                  {entry.license}
                </span>
                {entry.license_url && (
                  <a
                    href={entry.license_url}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="text-blue-700 hover:underline"
                  >
                    license details →
                  </a>
                )}
                <span className="text-zinc-500">
                  schema: {entry.schema}
                </span>
              </div>
              <div className="flex items-center gap-3 pt-2 border-t border-zinc-100">
                {!state && (
                  <button
                    type="button"
                    onClick={() => startDownload(entry)}
                    disabled={!api}
                    className="rounded-md bg-blue-600 text-white px-3 py-1.5 text-sm disabled:bg-zinc-300"
                  >
                    Download
                  </button>
                )}
                {isRunning && (
                  <span className="text-sm text-zinc-600">
                    Downloading… (first-time download fetches the full
                    dataset from Hugging Face; subsequent downloads use
                    the local HF cache.)
                  </span>
                )}
                {isDone && (
                  <>
                    <span className="text-sm text-emerald-800">
                      Saved {state.rows_kept?.toLocaleString()} of{" "}
                      {state.rows_loaded?.toLocaleString()} rows
                    </span>
                    <button
                      type="button"
                      onClick={() => useInTrain(entry)}
                      className="rounded-md bg-emerald-600 text-white px-3 py-1.5 text-sm hover:bg-emerald-700"
                    >
                      Use in Train →
                    </button>
                  </>
                )}
                {state?.status === "failed" && (
                  <>
                    <span className="text-sm text-red-700">
                      {state.error}
                    </span>
                    <button
                      type="button"
                      onClick={() => startDownload(entry)}
                      className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-50"
                    >
                      Retry
                    </button>
                  </>
                )}
              </div>
              {isDone && state.path && (
                <div className="text-xs font-mono text-zinc-500 break-all">
                  {state.path}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
