import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { useState } from "react";

import { useApiClient } from "../api/hooks";
import {
  type AppSettings,
  type BackendPref,
  loadSettings,
  saveSettings,
} from "../state/settings";

const BACKEND_OPTS: { value: BackendPref; label: string; help: string }[] = [
  { value: "auto", label: "Auto-detect", help: "Pick the first trainable device the probe returns." },
  { value: "cuda", label: "CUDA (NVIDIA)", help: "Prefer an NVIDIA GPU when present." },
  { value: "mlx", label: "MLX (Apple Silicon)", help: "Prefer the Apple Silicon device." },
];

const FORMAT_OPTS: { value: AppSettings["defaultDatasetFormat"]; label: string }[] = [
  { value: "jsonl_chat", label: "JSONL chat" },
  { value: "jsonl_chat_vision", label: "JSONL chat with images" },
  { value: "csv", label: "CSV" },
  { value: "text_dir", label: "Folder of .txt files" },
  { value: "hf_hub", label: "Hugging Face Hub" },
];

export function Settings() {
  const [draft, setDraft] = useState<AppSettings>(() => loadSettings());
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function pickOutputDir() {
    const result = await open({ multiple: false, directory: true });
    if (typeof result === "string") {
      setDraft((d) => ({ ...d, defaultOutputDir: result }));
    }
  }

  async function handleSave() {
    setError(null);
    saveSettings(draft);
    try {
      // Mirror to a config file the Rust side reads at next launch so it can
      // pass LLM_CHAIN_RUNS_DIR to the sidecar. localStorage on its own can't
      // reach the sidecar process.
      await invoke("save_desktop_settings", { settings: draft });
    } catch (e) {
      // Non-fatal: localStorage still holds the in-app defaults so backend /
      // dataset format prefs work. Output dir won't take effect.
      setError(`Saved in app, but couldn't write the launcher config: ${String(e)}`);
    }
    setSavedAt(Date.now());
  }

  return (
    <div className="p-6 space-y-6 max-w-2xl">
      <header>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-sm text-zinc-500">
          Defaults applied to new runs. Saved on this machine only.
        </p>
      </header>

      <section className="space-y-2">
        <label className="block text-sm font-medium">Default backend</label>
        <select
          value={draft.defaultBackend}
          onChange={(e) =>
            setDraft((d) => ({ ...d, defaultBackend: e.target.value as BackendPref }))
          }
          className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
        >
          {BACKEND_OPTS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-zinc-500">
          {BACKEND_OPTS.find((o) => o.value === draft.defaultBackend)?.help}
        </p>
      </section>

      <section className="space-y-2">
        <label className="block text-sm font-medium">Default dataset format</label>
        <select
          value={draft.defaultDatasetFormat}
          onChange={(e) =>
            setDraft((d) => ({
              ...d,
              defaultDatasetFormat: e.target.value as AppSettings["defaultDatasetFormat"],
            }))
          }
          className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
        >
          {FORMAT_OPTS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-zinc-500">
          Pre-selected on the Dataset screen. You can still change it per run.
        </p>
      </section>

      <section className="space-y-2">
        <label className="block text-sm font-medium">Default output directory</label>
        <div className="flex gap-2">
          <input
            value={draft.defaultOutputDir}
            readOnly
            placeholder="~/.llm-chain/runs"
            className="flex-1 rounded-md border border-zinc-300 px-3 py-2 text-sm font-mono"
          />
          <button
            type="button"
            onClick={pickOutputDir}
            className="rounded-md border border-zinc-300 px-4 py-2 text-sm hover:bg-zinc-50"
          >
            Browse…
          </button>
          {draft.defaultOutputDir && (
            <button
              type="button"
              onClick={() => setDraft((d) => ({ ...d, defaultOutputDir: "" }))}
              className="rounded-md border border-zinc-300 px-3 py-2 text-sm hover:bg-zinc-50"
            >
              Clear
            </button>
          )}
        </div>
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
          The sidecar reads <code>LLM_CHAIN_RUNS_DIR</code> at startup, so a new
          path takes effect <strong>after you restart the app</strong>. Existing
          runs stay where they are.
        </p>
      </section>

      <section className="space-y-2">
        <label className="flex items-start gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={draft.allowRestrictedModels}
            onChange={(e) =>
              setDraft((d) => ({ ...d, allowRestrictedModels: e.target.checked }))
            }
            className="mt-1"
          />
          <span>
            <span className="block text-sm font-medium">
              Allow restricted-license models (Llama, Gemma, DeepSeek)
            </span>
            <span className="block text-xs text-zinc-500 mt-0.5">
              Surfaces models with use-policy restrictions in the picker. Each
              entry shows its license caveat — review before downloading. You
              still need to accept the model's terms on Hugging Face.
            </span>
          </span>
        </label>
      </section>

      <CleanupSection />

      {error && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
          {error}
        </div>
      )}

      <div className="flex items-center gap-3 pt-2 border-t border-zinc-200">
        <button
          type="button"
          onClick={handleSave}
          className="rounded-md bg-blue-600 text-white px-5 py-2 text-sm font-medium"
        >
          Save
        </button>
        {savedAt && (
          <span className="text-sm text-green-700">Saved.</span>
        )}
      </div>
    </div>
  );
}

/**
 * Manual disk-cleanup pane. The user picks an age cutoff + statuses
 * and we delete every matching run via POST /api/maintenance/cleanup.
 *
 * Kept as a separate component so its API state doesn't interleave
 * with the Settings form's localStorage state — Settings is purely
 * client-side and synchronous, this talks to the sidecar.
 */
function CleanupSection() {
  const api = useApiClient();
  const [days, setDays] = useState(7);
  const [statuses, setStatuses] = useState<{
    failed: boolean;
    canceled: boolean;
    succeeded: boolean;
  }>({ failed: true, canceled: true, succeeded: false });
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ count: number; bytes: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  function selectedStatuses(): ("failed" | "canceled" | "succeeded")[] {
    const out: ("failed" | "canceled" | "succeeded")[] = [];
    if (statuses.failed) out.push("failed");
    if (statuses.canceled) out.push("canceled");
    if (statuses.succeeded) out.push("succeeded");
    return out;
  }

  async function applyNow() {
    if (!api) return;
    const picked = selectedStatuses();
    if (picked.length === 0) {
      setErr("Pick at least one status to clean.");
      return;
    }
    if (
      picked.includes("succeeded") &&
      !window.confirm(
        "You're about to delete SUCCEEDED runs (the ones with trained adapters). Are you sure?",
      )
    ) {
      return;
    }
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const r = await api.cleanupRuns({
        older_than_days: days,
        statuses: picked,
      });
      setResult({ count: r.deleted_count, bytes: r.freed_bytes });
    } catch (e) {
      setErr(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  }

  function fmtBytes(n: number): string {
    if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
    if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
    if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${n} B`;
  }

  return (
    <section className="space-y-3">
      <header>
        <h2 className="text-sm font-medium">Clean up old runs</h2>
        <p className="text-xs text-zinc-500">
          Bulk-delete runs older than the cutoff. Reveals back the disk
          space their adapters and intermediate checkpoints are
          holding. Active runs (pending / running) are never affected.
        </p>
      </header>
      <div className="grid grid-cols-1 sm:grid-cols-[160px_1fr] gap-3 items-start">
        <label className="text-sm space-y-1 block">
          <span className="block text-xs text-zinc-600">Older than (days)</span>
          <input
            type="number"
            min={0}
            value={days}
            onChange={(e) => {
              const n = parseFloat(e.target.value);
              if (Number.isFinite(n) && n >= 0) setDays(n);
            }}
            className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
          />
        </label>
        <div className="space-y-1.5">
          <span className="block text-xs text-zinc-600">Statuses to delete</span>
          {(["failed", "canceled", "succeeded"] as const).map((s) => (
            <label key={s} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={statuses[s]}
                onChange={(e) =>
                  setStatuses((p) => ({ ...p, [s]: e.target.checked }))
                }
              />
              <span>
                {s}
                {s === "succeeded" && (
                  <span className="ml-1 text-xs text-amber-700">
                    (deletes trained adapters!)
                  </span>
                )}
              </span>
            </label>
          ))}
        </div>
      </div>
      <button
        type="button"
        onClick={applyNow}
        disabled={busy}
        className="rounded-md border border-zinc-300 px-4 py-2 text-sm hover:bg-zinc-50 disabled:opacity-50"
      >
        {busy ? "Cleaning…" : "Apply now"}
      </button>
      {result && (
        <div className="text-sm text-green-700 bg-green-50 border border-green-200 rounded p-2">
          Deleted {result.count} run{result.count === 1 ? "" : "s"} ·
          freed {fmtBytes(result.bytes)}.
        </div>
      )}
      {err && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
          {err}
        </div>
      )}
    </section>
  );
}
