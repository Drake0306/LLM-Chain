import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { useState } from "react";

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
