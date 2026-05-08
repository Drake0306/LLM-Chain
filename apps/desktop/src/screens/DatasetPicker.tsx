import { open } from "@tauri-apps/plugin-dialog";
import { useEffect, useMemo, useState } from "react";

import type { DatasetPreview } from "../api/client";
import { useApiClient } from "../api/hooks";
import {
  FORMAT_ORDER,
  FORMAT_SPECS,
  supportedFormats,
} from "../state/datasetSupport";
import type { DatasetChoice } from "../state/selection";
import { useSelection } from "../state/selection";
import { loadSettings } from "../state/settings";

export function DatasetPicker() {
  const api = useApiClient();
  const { dataset, setDataset, model } = useSelection();
  // Memoise so the resnap effect doesn't fire on every render — the
  // helper returns a fresh array each call, which would otherwise make
  // [allowedFormats] a new identity every render.
  const allowedFormats = useMemo(() => supportedFormats(model), [model]);

  // Default the format to one the selected model can actually train on.
  // If a stale selection (e.g. from a different model) carries a format
  // the current model doesn't support, fall back to the first allowed
  // format instead of leaving the user with a "you can't train on this"
  // mismatch hidden in the dropdown.
  const initialFormat: DatasetChoice["format"] = (() => {
    if (dataset?.format && allowedFormats.includes(dataset.format)) return dataset.format;
    const settingsDefault = loadSettings().defaultDatasetFormat;
    if (allowedFormats.includes(settingsDefault)) return settingsDefault;
    return allowedFormats[0];
  })();

  const [format, setFormat] = useState<DatasetChoice["format"]>(initialFormat);
  const [path, setPath] = useState<string>(dataset?.path ?? "");
  const [hfId, setHfId] = useState<string>(dataset?.hf_id ?? "");
  const [textColumn, setTextColumn] = useState<string>(dataset?.text_column ?? "text");
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewing, setPreviewing] = useState(false);

  // If the user navigates back to this page after changing models, the
  // current format may no longer be allowed. Snap it back to a supported
  // option so the picker can never be in an invalid state.
  useEffect(() => {
    if (!allowedFormats.includes(format)) {
      setFormat(allowedFormats[0]);
    }
  }, [allowedFormats, format]);

  const spec = FORMAT_SPECS[format];

  async function pickFile() {
    const result = await open({
      multiple: false,
      directory: spec.filePicker === "directory",
      filters:
        spec.filePicker === "jsonl"
          ? [{ name: "JSONL", extensions: ["jsonl"] }]
          : spec.filePicker === "csv"
            ? [{ name: "CSV", extensions: ["csv"] }]
            : undefined,
    });
    if (typeof result === "string") {
      setPath(result);
    }
  }

  function commit() {
    const next: DatasetChoice = { format };
    if (format === "hf_hub") next.hf_id = hfId.trim();
    else next.path = path.trim();
    if (format === "csv") next.text_column = textColumn.trim();
    setDataset(next);
  }

  const ready =
    format === "hf_hub" ? hfId.trim().length > 0 : path.trim().length > 0;

  async function handlePreview() {
    if (!api || !ready) return;
    setPreviewError(null);
    setPreviewing(true);
    try {
      // HF Hub previews are skipped — the loader would download the
      // entire dataset to give us 3 rows, which isn't what the user
      // signed up for when clicking "Preview".
      if (format === "hf_hub") {
        setPreviewError(
          "HF Hub preview isn't supported — the loader would have to fetch the dataset first. Just pick the dataset and start training; format errors will surface there.",
        );
        return;
      }
      const result = await api.previewDataset({
        dataset_path: path.trim(),
        dataset_format: format,
        text_column: format === "csv" ? textColumn.trim() : undefined,
        limit: 3,
      });
      setPreview(result);
    } catch (e) {
      setPreviewError(String((e as Error).message ?? e));
      setPreview(null);
    } finally {
      setPreviewing(false);
    }
  }

  // Clear stale preview when the user changes path / format / column.
  useEffect(() => {
    setPreview(null);
    setPreviewError(null);
  }, [format, path, hfId, textColumn]);

  return (
    <div className="p-6 grid grid-cols-1 xl:grid-cols-[minmax(0,28rem)_1fr] gap-8">
      <div className="space-y-6 max-w-2xl">
        <header>
          <h1 className="text-2xl font-semibold">Pick a Dataset</h1>
          <p className="text-sm text-zinc-500">{spec.help}</p>
          {model && (
            <p className="mt-2 text-xs text-zinc-600 bg-zinc-50 border border-zinc-200 rounded p-2 leading-relaxed">
              Showing only formats compatible with{" "}
              <span className="font-medium">{model.name}</span>
              {allowedFormats.length < FORMAT_ORDER.length && (
                <>
                  {" — "}
                  {FORMAT_ORDER.length - allowedFormats.length} other format
                  {FORMAT_ORDER.length - allowedFormats.length === 1 ? "" : "s"}{" "}
                  hidden because this model doesn't support them.
                </>
              )}
            </p>
          )}
          {!model && (
            <p className="mt-2 text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 leading-relaxed">
              No model picked yet — pick one on the Models page first to filter
              the format list to what's actually trainable.
            </p>
          )}
        </header>

        <div className="space-y-2">
          <label className="block text-sm font-medium">Format</label>
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value as DatasetChoice["format"])}
            className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
          >
            {allowedFormats.map((f) => (
              <option key={f} value={f}>
                {FORMAT_SPECS[f].label}
              </option>
            ))}
          </select>
        </div>

        {format === "hf_hub" ? (
          <div className="space-y-2">
            <label className="block text-sm font-medium">Hugging Face dataset ID</label>
            <input
              value={hfId}
              onChange={(e) => setHfId(e.target.value)}
              placeholder="acme/dataset-name"
              className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
            />
          </div>
        ) : (
          <div className="space-y-2">
            <label className="block text-sm font-medium">
              {format === "text_dir" ? "Folder" : "File"}
            </label>
            <div className="flex gap-2">
              <input
                value={path}
                readOnly
                placeholder={format === "text_dir" ? "/path/to/folder" : "/path/to/file"}
                className="flex-1 rounded-md border border-zinc-300 px-3 py-2 text-sm font-mono"
              />
              <button
                type="button"
                onClick={pickFile}
                className="rounded-md border border-zinc-300 px-4 py-2 text-sm hover:bg-zinc-50"
              >
                Browse…
              </button>
            </div>
          </div>
        )}

        {format === "csv" && (
          <div className="space-y-2">
            <label className="block text-sm font-medium">Text column</label>
            <input
              value={textColumn}
              onChange={(e) => setTextColumn(e.target.value)}
              className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm font-mono"
            />
          </div>
        )}

        {ready && format !== "hf_hub" && (
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handlePreview}
              disabled={previewing}
              className="text-sm rounded-md border border-zinc-300 px-3 py-1.5 hover:bg-zinc-50 disabled:opacity-50"
            >
              {previewing ? "Reading…" : "Preview my data"}
            </button>
            <span className="text-xs text-zinc-500">
              Reads the first 3 rows through the same parser the trainer
              uses — catches format mistakes before clicking Train.
            </span>
          </div>
        )}

        <div className="flex items-center justify-between pt-2 border-t border-zinc-200">
          <div className="text-sm text-zinc-500">
            {dataset
              ? `Saved: ${dataset.format} → ${dataset.path ?? dataset.hf_id}`
              : "Not yet saved."}
          </div>
          <button
            type="button"
            onClick={commit}
            disabled={!ready}
            className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm disabled:bg-zinc-300"
          >
            Use this dataset
          </button>
        </div>
      </div>

      <FormatExamplePanel
        format={format}
        preview={preview}
        previewError={previewError}
      />
    </div>
  );
}

/**
 * Right-rail example + live preview. Sticks while the user scrolls the
 * form so they can always see what shape their data needs.
 *
 * When the user clicks "Preview my data" we replace the synthetic
 * example with the loader's actual output for their file. Format
 * errors show inline below — the user fixes the file, re-clicks
 * Preview, and only when it parses cleanly do they move on to Train.
 */
function FormatExamplePanel({
  format,
  preview,
  previewError,
}: {
  format: DatasetChoice["format"];
  preview: DatasetPreview | null;
  previewError: string | null;
}) {
  const spec = FORMAT_SPECS[format];
  return (
    <aside className="xl:sticky xl:top-6 self-start space-y-3 min-w-0">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-semibold">
          {preview ? "Your data" : "Example"}
        </h2>
        <span className="text-xs text-zinc-500">
          {preview
            ? `${preview.shown} of ${preview.row_count} row${preview.row_count === 1 ? "" : "s"}`
            : spec.label}
        </span>
      </div>
      <p className="text-xs text-zinc-600 leading-relaxed">{spec.help}</p>

      {previewError && (
        <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded p-2 leading-relaxed whitespace-pre-wrap">
          {previewError}
        </div>
      )}

      <pre className="text-xs leading-relaxed bg-zinc-900 text-zinc-100 rounded-lg p-4 overflow-x-auto whitespace-pre">
        {preview
          ? preview.rows.map((r) => JSON.stringify(r, null, 2)).join("\n\n")
          : spec.example}
      </pre>

      {preview && preview.row_count < 2 && (
        <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 leading-relaxed">
          Only {preview.row_count} row found. Training needs at least 2
          so the train and validation splits don't overlap — add another
          row before clicking Train.
        </div>
      )}

      <FormatTips format={format} />
    </aside>
  );
}

function FormatTips({ format }: { format: DatasetChoice["format"] }) {
  // Format-specific gotchas the loader / trainer enforce. Surfacing them
  // here lets the user fix the data BEFORE clicking Train and seeing a
  // 30-line stack trace.
  const tips: Record<DatasetChoice["format"], string[]> = {
    jsonl_chat: [
      "One JSON object per line — no commas between rows, no outer array.",
      "Each row needs a 'messages' array; each message needs 'role' and 'content'.",
      "Trailing blank lines are ignored. UTF-8 only.",
    ],
    jsonl_chat_vision: [
      "Image paths can be relative — they're resolved against the JSONL's folder.",
      "Each image must exist on disk; the loader validates before training starts.",
      "Mix image and text parts freely inside one user message.",
    ],
    csv: [
      "Header row is required.",
      "Set 'Text column' below to whichever column holds the training text.",
      "Quoted multi-line cells are supported (standard CSV rules).",
    ],
    text_dir: [
      "Every .txt under the folder becomes one example, including subdirs.",
      "Other extensions (.md, .json) are ignored.",
      "Each file's full contents are one row — split big docs first if you want shorter examples.",
    ],
    hf_hub: [
      "Public datasets work without auth. Private datasets need huggingface-cli login.",
      "The loader auto-detects 'text', 'content', or 'input' columns.",
      "Set 'Text column' if your dataset uses a different field name.",
    ],
  };
  return (
    <ul className="text-xs text-zinc-600 leading-relaxed space-y-1.5 list-disc pl-4">
      {tips[format].map((t, i) => (
        <li key={i}>{t}</li>
      ))}
    </ul>
  );
}
