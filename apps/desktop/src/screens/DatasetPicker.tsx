import { open } from "@tauri-apps/plugin-dialog";
import { useState } from "react";

import type { DatasetChoice } from "../state/selection";
import { useSelection } from "../state/selection";
import { loadSettings } from "../state/settings";

const FORMATS: { value: DatasetChoice["format"]; label: string; help: string }[] = [
  {
    value: "jsonl_chat",
    label: "JSONL chat",
    help: 'One row per JSON object: {"messages":[{"role":"user","content":"…"},…]}',
  },
  {
    value: "jsonl_chat_vision",
    label: "JSONL chat with images",
    help: 'OpenAI-style content arrays. {"messages":[{"role":"user","content":[{"type":"image","path":"./img.png"},{"type":"text","text":"…"}]},…]}. Image paths are resolved relative to the JSONL file.',
  },
  { value: "csv", label: "CSV", help: "Pick a column that holds the text to train on." },
  { value: "text_dir", label: "Folder of .txt files", help: "Each file becomes one row." },
  { value: "hf_hub", label: "Hugging Face Hub", help: "Datasets ID like 'allenai/c4'." },
];

export function DatasetPicker() {
  const { dataset, setDataset } = useSelection();
  const [format, setFormat] = useState<DatasetChoice["format"]>(
    dataset?.format ?? loadSettings().defaultDatasetFormat,
  );
  const [path, setPath] = useState<string>(dataset?.path ?? "");
  const [hfId, setHfId] = useState<string>(dataset?.hf_id ?? "");
  const [textColumn, setTextColumn] = useState<string>(dataset?.text_column ?? "text");

  const help = FORMATS.find((f) => f.value === format)?.help;

  async function pickFile() {
    const result = await open({
      multiple: false,
      directory: format === "text_dir",
      filters:
        format === "jsonl_chat" || format === "jsonl_chat_vision"
          ? [{ name: "JSONL", extensions: ["jsonl"] }]
          : format === "csv"
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

  return (
    <div className="p-6 space-y-6 max-w-2xl">
      <header>
        <h1 className="text-2xl font-semibold">Pick a Dataset</h1>
        <p className="text-sm text-zinc-500">{help}</p>
      </header>

      <div className="space-y-2">
        <label className="block text-sm font-medium">Format</label>
        <select
          value={format}
          onChange={(e) => setFormat(e.target.value as DatasetChoice["format"])}
          className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
        >
          {FORMATS.map((f) => (
            <option key={f.value} value={f.value}>
              {f.label}
            </option>
          ))}
        </select>
        <details className="text-xs text-zinc-600">
          <summary className="cursor-pointer">Which format should I pick?</summary>
          <ul className="mt-2 space-y-1 list-disc pl-4 leading-relaxed">
            <li><span className="font-medium">JSONL chat</span> — most common. One conversation per line; works with any chat / instruct model.</li>
            <li><span className="font-medium">JSONL chat with images</span> — same shape, but content is a list of parts so user turns can include images. Pair with a vision-language model (Qwen2-VL).</li>
            <li><span className="font-medium">CSV</span> — pick this when you have a spreadsheet of plain-text examples. You name the column to train on.</li>
            <li><span className="font-medium">Folder of .txt files</span> — each .txt becomes one example. Useful for ad-hoc local corpora.</li>
            <li><span className="font-medium">Hugging Face Hub</span> — pulls a public dataset by ID (e.g. <code className="font-mono">HuggingFaceH4/no_robots</code>). No download step.</li>
          </ul>
        </details>
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
  );
}
