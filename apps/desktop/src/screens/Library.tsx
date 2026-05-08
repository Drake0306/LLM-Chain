import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import type { Run } from "../api/client";
import { useApiClient } from "../api/hooks";

/**
 * Adapter library: every SUCCEEDED run, sortable, with quick actions.
 *
 * The /runs page lists everything (pending, failed, succeeded) for
 * inspection; this is the "what have I actually built?" view —
 * filtered to runs that produced an adapter, with size on disk so
 * users can spot accidental disk-fill. Actions delegate to the same
 * endpoints RunDetail uses (Reveal, Playground, Delete) instead of
 * duplicating the UI.
 */

type SortKey = "created" | "size" | "model";

function formatBytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  // Local time, "MMM d, HH:mm" — short enough to fit in a row.
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function Library() {
  const api = useApiClient();
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("created");
  const [reverse, setReverse] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  // Set of run ids ticked for bulk action. Cleared on refresh so a
  // delete that completes doesn't leave selections pointing at runs
  // that no longer exist.
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkBusy, setBulkBusy] = useState(false);

  function refresh() {
    if (!api) return;
    setError(null);
    api.listRuns()
      .then((r) => {
        const succeeded = r.runs.filter((x) => x.status === "succeeded");
        setRuns(succeeded);
        // Drop selected ids that no longer exist — common after a
        // bulk delete; otherwise the count stays misleading.
        const stillThere = new Set(succeeded.map((x) => x.id));
        setSelected((prev) => {
          const next = new Set<string>();
          prev.forEach((id) => {
            if (stillThere.has(id)) next.add(id);
          });
          return next;
        });
      })
      .catch((e: unknown) => setError(String((e as Error).message ?? e)));
  }

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleAll(visible: Run[]) {
    setSelected((prev) => {
      // If all visible are already selected, clear; otherwise select
      // all visible (preserving any selection from outside the
      // current sort/filter — there isn't filtering yet, but be
      // forward-compatible).
      const allSelected = visible.every((r) => prev.has(r.id));
      if (allSelected) {
        const next = new Set(prev);
        visible.forEach((r) => next.delete(r.id));
        return next;
      }
      const next = new Set(prev);
      visible.forEach((r) => next.add(r.id));
      return next;
    });
  }

  async function deleteSelected() {
    if (!api || selected.size === 0) return;
    if (
      !window.confirm(
        `Delete ${selected.size} run${selected.size === 1 ? "" : "s"} permanently? Cannot be undone.`,
      )
    )
      return;
    setBulkBusy(true);
    setActionError(null);
    // Sequential — bulk-deleting in parallel could swamp the
    // sidecar's RunStore writes and starve other requests. The
    // ergonomic loss vs parallel is small; users delete dozens at
    // most.
    const failures: { id: string; reason: string }[] = [];
    for (const id of selected) {
      try {
        await api.deleteRun(id);
      } catch (e) {
        failures.push({ id, reason: String((e as Error).message ?? e) });
      }
    }
    setBulkBusy(false);
    if (failures.length > 0) {
      setActionError(
        `Deleted ${selected.size - failures.length} of ${selected.size}. ` +
          `Failures: ${failures.map((f) => `${f.id} (${f.reason})`).join("; ")}`,
      );
    }
    refresh();
  }

  useEffect(() => {
    refresh();
    // Disabled exhaustive-deps: refresh closes over `api`, but we
    // only want it to fire on the api binding becoming available.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  const sorted = useMemo(() => {
    if (!runs) return null;
    const copy = runs.slice();
    copy.sort((a, b) => {
      let cmp = 0;
      if (sortKey === "created") {
        cmp = a.created_at.localeCompare(b.created_at);
      } else if (sortKey === "model") {
        cmp = a.config.model_id.localeCompare(b.config.model_id);
      } else if (sortKey === "size") {
        // Treat null sizes as the smallest so they sink in
        // descending-by-size mode (the default for users hunting
        // disk hogs).
        const av = a.adapter_size_bytes ?? -1;
        const bv = b.adapter_size_bytes ?? -1;
        cmp = av - bv;
      }
      return reverse ? -cmp : cmp;
    });
    return copy;
  }, [runs, sortKey, reverse]);

  async function handleDelete(run: Run) {
    if (!api) return;
    if (!window.confirm(
      `Delete adapter from run ${run.id}? Files on disk go away. Cannot be undone.`,
    )) return;
    setBusyId(run.id);
    setActionError(null);
    try {
      await api.deleteRun(run.id);
      refresh();
    } catch (e) {
      setActionError(String((e as Error).message ?? e));
    } finally {
      setBusyId(null);
    }
  }

  function header(label: string, key: SortKey) {
    const active = sortKey === key;
    const arrow = active ? (reverse ? " ↑" : " ↓") : "";
    return (
      <button
        type="button"
        onClick={() => {
          if (active) setReverse(!reverse);
          else {
            setSortKey(key);
            // Sensible default: size descending (biggest first), date
            // descending (newest first), model ascending (alpha).
            setReverse(key !== "model");
          }
        }}
        className={`text-left text-xs uppercase tracking-wide ${
          active ? "text-zinc-900" : "text-zinc-500 hover:text-zinc-700"
        }`}
      >
        {label}
        {arrow}
      </button>
    );
  }

  if (error) {
    return (
      <div className="p-6 space-y-2">
        <h1 className="text-2xl font-semibold">Library</h1>
        <pre className="text-xs bg-red-50 border border-red-200 rounded p-3 whitespace-pre-wrap">
          {error}
        </pre>
      </div>
    );
  }
  if (!sorted) {
    return <div className="p-6 text-zinc-500">Loading library…</div>;
  }
  if (sorted.length === 0) {
    return (
      <div className="p-6 space-y-2 text-zinc-500">
        <h1 className="text-2xl font-semibold text-zinc-900">Library</h1>
        <p className="text-sm">
          No trained adapters yet — finish a successful run on the{" "}
          <Link to="/train" className="underline">Train</Link> page and it'll
          show up here.
        </p>
      </div>
    );
  }

  const totalSize = sorted.reduce(
    (sum, r) => sum + (r.adapter_size_bytes ?? 0),
    0,
  );

  return (
    <div className="p-6 space-y-4">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Library</h1>
          <p className="text-sm text-zinc-500">
            {sorted.length} trained adapter{sorted.length === 1 ? "" : "s"} ·{" "}
            {formatBytes(totalSize)} total on disk
            {sorted.length >= 2 && (
              <>
                {" · "}
                <Link
                  to="/compare/prompts"
                  className="text-blue-700 hover:underline"
                >
                  Compare two on prompts →
                </Link>
              </>
            )}
          </p>
        </div>
        {selected.size > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">
              {selected.size} selected
            </span>
            <button
              type="button"
              onClick={deleteSelected}
              disabled={bulkBusy}
              className="text-xs px-3 py-1.5 rounded-md border border-red-200 text-red-700 hover:bg-red-50 disabled:opacity-50"
            >
              {bulkBusy ? "Deleting…" : `Delete ${selected.size}`}
            </button>
            <button
              type="button"
              onClick={() => setSelected(new Set())}
              className="text-xs text-zinc-500 hover:text-zinc-700"
            >
              clear
            </button>
          </div>
        )}
      </header>

      {actionError && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
          {actionError}
        </div>
      )}

      <div className="border border-zinc-200 rounded-lg overflow-hidden">
        <div className="grid grid-cols-[24px_1fr_140px_120px_220px] gap-3 px-4 py-2 bg-zinc-50 border-b border-zinc-200 items-center">
          <input
            type="checkbox"
            checked={
              sorted.length > 0 && sorted.every((r) => selected.has(r.id))
            }
            // Indeterminate when *some* but not all rows are
            // selected. Set imperatively because React doesn't
            // expose it on the input element.
            ref={(el) => {
              if (el) {
                const all = sorted.length > 0 && sorted.every((r) => selected.has(r.id));
                const any = sorted.some((r) => selected.has(r.id));
                el.indeterminate = any && !all;
              }
            }}
            onChange={() => toggleAll(sorted)}
            title={
              sorted.every((r) => selected.has(r.id))
                ? "Clear selection"
                : "Select all visible"
            }
          />
          {header("Model · Run", "model")}
          {header("Created", "created")}
          {header("Size", "size")}
          <span className="text-xs uppercase tracking-wide text-zinc-500">
            Actions
          </span>
        </div>
        <ul className="divide-y divide-zinc-200">
          {sorted.map((r) => {
            const busy = busyId === r.id;
            const checked = selected.has(r.id);
            return (
              <li
                key={r.id}
                className="grid grid-cols-[24px_1fr_140px_120px_220px] gap-3 px-4 py-3 items-center"
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggle(r.id)}
                  title="Select for bulk action"
                />
                <div className="min-w-0">
                  <div className="text-sm truncate" title={r.config.model_id}>
                    {r.config.model_id}
                  </div>
                  <Link
                    to={`/runs/${r.id}`}
                    className="font-mono text-xs text-zinc-500 hover:underline"
                  >
                    {r.id} · {r.config.technique?.toUpperCase() ?? "?"} · {r.config.backend}
                  </Link>
                  {/* Dataset trail — useful for "which fine-tune was on
                      which data" disambiguation when models repeat across
                      runs. Truncates with the file path's tail visible
                      since that's usually the distinguishing bit. */}
                  <div
                    className="text-xs text-zinc-500 truncate font-mono"
                    title={r.config.dataset_path}
                  >
                    {r.config.dataset_format ?? "?"} ·{" "}
                    {r.config.dataset_path
                      ? r.config.dataset_path.split("/").slice(-2).join("/")
                      : "—"}
                  </div>
                </div>
                <div className="text-xs text-zinc-600 tabular-nums">
                  {formatDate(r.created_at)}
                </div>
                <div className="text-xs font-mono text-zinc-600 tabular-nums">
                  {formatBytes(r.adapter_size_bytes)}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  <Link
                    to={`/runs/${r.id}/play`}
                    className="text-xs px-2 py-1 rounded border border-blue-200 text-blue-700 hover:bg-blue-50"
                  >
                    Play
                  </Link>
                  {r.output_dir && (
                    <button
                      type="button"
                      onClick={async () => {
                        try {
                          setActionError(null);
                          await revealItemInDir(r.output_dir as string);
                        } catch (e) {
                          setActionError(String((e as Error).message ?? e));
                        }
                      }}
                      className="text-xs px-2 py-1 rounded border border-zinc-300 text-zinc-700 hover:bg-zinc-50"
                    >
                      Reveal
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => handleDelete(r)}
                    disabled={busy}
                    className="text-xs px-2 py-1 rounded border border-red-200 text-red-700 hover:bg-red-50 disabled:opacity-50"
                  >
                    {busy ? "…" : "Delete"}
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
