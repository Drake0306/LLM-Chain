import { open } from "@tauri-apps/plugin-dialog";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import type { DatasetBuildResult } from "../api/client";
import { useApiClient } from "../api/hooks";
import { useSelection } from "../state/selection";
import {
  type InputFormat,
  type SchemaMapping,
  detectSchema,
  formatBytes,
  parseText,
} from "../state/workshop";

const PREVIEW_ROWS = 3;
const PASTE_PLACEHOLDER: Record<InputFormat, string> = {
  csv:
    "user,assistant\n" +
    'What is 2 + 2?,4\n' +
    'Who wrote Hamlet?,William Shakespeare.\n' +
    'Capital of France?,Paris.\n',
  tsv: "user\tassistant\nWhat is 2 + 2?\t4\nWho wrote Hamlet?\tWilliam Shakespeare.\n",
  jsonl:
    '{"messages":[{"role":"user","content":"What is 2 + 2?"},' +
    '{"role":"assistant","content":"4"}]}\n' +
    '{"messages":[{"role":"user","content":"Capital of France?"},' +
    '{"role":"assistant","content":"Paris."}]}\n',
};

export function DatasetWorkshop() {
  const api = useApiClient();
  const navigate = useNavigate();
  const { setDataset } = useSelection();

  const [format, setFormat] = useState<InputFormat>("csv");
  const [text, setText] = useState<string>(PASTE_PLACEHOLDER.csv);
  const [filePath, setFilePath] = useState<string | null>(null);
  const [name, setName] = useState<string>("");
  const [schema, setSchema] = useState<SchemaMapping>({
    target: "chat",
    user_field: "user",
    assistant_field: "assistant",
    passthrough_chat: false,
  });
  const [autoDetect, setAutoDetect] = useState(true);

  // Cleaning toggles. role_balance defaults off when the user is in
  // passthrough mode because role_balance trims rows that don't start
  // user→assistant, and the server's already-chat input often has
  // legitimate system-prompt rows we don't want to throw out.
  const [dropEmpty, setDropEmpty] = useState(true);
  const [dedupe, setDedupe] = useState(true);
  const [roleBalance, setRoleBalance] = useState(true);
  const [maxChars, setMaxChars] = useState<number | null>(null);

  const [building, setBuilding] = useState(false);
  const [result, setResult] = useState<DatasetBuildResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Live parse → preview. Debouncing isn't necessary at the sizes the
  // textarea reasonably holds (a few hundred KB), but if it grows
  // sluggish on big pastes, switch to a useDeferredValue here.
  const parsed = useMemo(() => parseText(text, format), [text, format]);

  // Auto-update schema mapping when the parsed columns change. Once the
  // user manually edits a field, autoDetect flips off so we don't keep
  // overwriting their choice.
  useEffect(() => {
    if (!autoDetect) return;
    if (parsed.rows.length === 0) return;
    const detected = detectSchema(parsed.rows);
    setSchema(detected);
  }, [parsed.rows, autoDetect]);

  function setSchemaField<K extends keyof SchemaMapping>(
    key: K,
    value: SchemaMapping[K],
  ) {
    setAutoDetect(false);
    setSchema((s) => ({ ...s, [key]: value }));
  }

  const columns = parsed.columns;
  const rowCount = parsed.rows.length;
  const sampleRows = parsed.rows.slice(0, PREVIEW_ROWS);

  async function pickFile() {
    const ext =
      format === "csv" ? ["csv"] : format === "tsv" ? ["tsv", "tab"] : ["jsonl"];
    const result = await open({
      multiple: false,
      directory: false,
      filters: [{ name: format.toUpperCase(), extensions: ext }],
    });
    if (typeof result === "string") {
      setFilePath(result);
    }
  }

  async function handleBuild() {
    if (!api) return;
    setError(null);
    setResult(null);
    setBuilding(true);
    try {
      const body = {
        input_format: format,
        target: schema.target,
        user_field: schema.user_field,
        assistant_field: schema.assistant_field,
        prompt_field: schema.prompt_field,
        completion_field: schema.completion_field,
        passthrough_chat: schema.passthrough_chat,
        drop_empty: dropEmpty,
        dedupe,
        role_balance: roleBalance,
        max_chars: maxChars,
        name: name.trim() || undefined,
        ...(filePath
          ? { source_path: filePath }
          : { raw_text: text }),
      };
      const r = await api.buildDataset(body);
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBuilding(false);
    }
  }

  function handleUseInTrain() {
    if (!result) return;
    setDataset({ format: "jsonl_chat", path: result.path });
    navigate("/train");
  }

  const stats = result?.stats;
  // Surface the active source. When a file is picked, its path
  // overrides the textarea entirely on submit; show this so the user
  // doesn't think their pasted text is being used.
  const usingFile = !!filePath;

  return (
    <div className="p-6 max-w-6xl space-y-6">
      <header className="flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Dataset Workshop</h1>
          <p className="text-sm text-zinc-500">
            Paste rows or upload a file, map columns onto chat fields, run
            cleaners, and save as JSONL the trainer can use directly.
          </p>
        </div>
        <Link
          to="/dataset"
          className="text-sm text-zinc-600 hover:text-zinc-900"
        >
          ← Back to Dataset picker
        </Link>
      </header>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* LEFT: source + schema + cleaners */}
        <div className="space-y-5">
          <div className="space-y-2">
            <label className="block text-sm font-medium">Input format</label>
            <div className="flex gap-2 text-sm">
              {(["csv", "tsv", "jsonl"] as InputFormat[]).map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => {
                    setFormat(f);
                    if (text === PASTE_PLACEHOLDER[format]) {
                      setText(PASTE_PLACEHOLDER[f]);
                    }
                    setAutoDetect(true);
                  }}
                  className={`px-3 py-1.5 rounded-md border ${
                    format === f
                      ? "bg-blue-600 text-white border-blue-600"
                      : "bg-white border-zinc-300 hover:bg-zinc-50"
                  }`}
                >
                  {f.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">
              Source
              <span className="ml-2 text-xs font-normal text-zinc-500">
                Paste rows OR upload a file (file wins on Build).
              </span>
            </label>
            <textarea
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                setAutoDetect(true);
              }}
              rows={10}
              spellCheck={false}
              className="w-full rounded-md border border-zinc-300 px-3 py-2 text-xs font-mono"
            />
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={pickFile}
                className="text-sm rounded-md border border-zinc-300 px-3 py-1.5 hover:bg-zinc-50"
              >
                {usingFile ? "Replace file…" : "Upload file…"}
              </button>
              {usingFile && (
                <button
                  type="button"
                  onClick={() => setFilePath(null)}
                  className="text-sm text-red-700 underline-offset-2 hover:underline"
                >
                  Use pasted text instead
                </button>
              )}
              <span className="text-xs text-zinc-500 truncate">
                {usingFile ? `Using: ${filePath}` : `Pasted: ${rowCount} row${rowCount === 1 ? "" : "s"}`}
              </span>
            </div>
            {parsed.error && !usingFile && (
              <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 leading-relaxed">
                {parsed.error}
              </div>
            )}
          </div>

          <SchemaSection
            schema={schema}
            columns={columns}
            format={format}
            onChange={setSchemaField}
            usingFile={usingFile}
          />

          <div className="space-y-2">
            <label className="block text-sm font-medium">Cleaners</label>
            <div className="space-y-2">
              <Toggle
                checked={dropEmpty}
                onChange={setDropEmpty}
                label="Drop empty rows"
                hint="Rows whose user or assistant content is blank after trimming."
              />
              <Toggle
                checked={dedupe}
                onChange={setDedupe}
                label="Deduplicate"
                hint="Drop rows whose chat content is byte-for-byte identical to one already kept."
              />
              <Toggle
                checked={roleBalance}
                onChange={setRoleBalance}
                label="Enforce user → assistant order"
                hint="Drop rows that start with assistant or have only one role; a leading system message is allowed."
              />
              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  checked={maxChars !== null}
                  onChange={(e) =>
                    setMaxChars(e.target.checked ? 4000 : null)
                  }
                />
                <span className="text-sm">Length limit (chars)</span>
                <input
                  type="number"
                  value={maxChars ?? ""}
                  disabled={maxChars === null}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    setMaxChars(Number.isFinite(v) && v > 0 ? v : null);
                  }}
                  className="w-28 rounded-md border border-zinc-300 px-2 py-1 text-sm disabled:bg-zinc-100"
                  min={1}
                />
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">
              Output filename
              <span className="ml-2 text-xs font-normal text-zinc-500">
                Saved to ~/.llm-chain/datasets/&lt;name&gt;.jsonl
              </span>
            </label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="my-customer-support-set"
              className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
            />
          </div>

          <div className="flex items-center gap-3 pt-2 border-t border-zinc-200">
            <button
              type="button"
              onClick={handleBuild}
              disabled={building || !api || (rowCount === 0 && !usingFile)}
              className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm disabled:bg-zinc-300"
            >
              {building ? "Building…" : "Build dataset"}
            </button>
            {error && (
              <span className="text-xs text-red-700 leading-relaxed whitespace-pre-wrap">
                {error}
              </span>
            )}
          </div>
        </div>

        {/* RIGHT: preview + result */}
        <aside className="space-y-4">
          <h2 className="text-sm font-semibold">Preview</h2>
          {sampleRows.length === 0 ? (
            <div className="text-xs text-zinc-500">
              {usingFile
                ? "File contents are read on Build."
                : "Type or paste rows on the left."}
            </div>
          ) : (
            <pre className="text-xs leading-relaxed bg-zinc-900 text-zinc-100 rounded-lg p-4 overflow-x-auto whitespace-pre">
              {sampleRows.map((r) => JSON.stringify(r, null, 2)).join("\n\n")}
            </pre>
          )}
          {sampleRows.length > 0 && rowCount > sampleRows.length && (
            <div className="text-xs text-zinc-500">
              Showing {sampleRows.length} of {rowCount} parsed rows.
            </div>
          )}

          {result && (
            <div className="space-y-2 rounded-md border border-emerald-200 bg-emerald-50 p-3">
              <div className="text-sm font-medium text-emerald-900">
                Built {stats!.output_rows} row{stats!.output_rows === 1 ? "" : "s"}
                {" "}
                <span className="text-emerald-700">
                  ({formatBytes(result.bytes_written)})
                </span>
              </div>
              <div className="text-xs text-emerald-900 font-mono break-all">
                {result.path}
              </div>
              <div className="text-xs text-emerald-900 leading-relaxed">
                Dropped:{" "}
                {stats!.dropped_empty} empty,{" "}
                {stats!.dropped_duplicate} duplicate,{" "}
                {stats!.dropped_role_violation} role,{" "}
                {stats!.dropped_length} length.
              </div>
              <button
                type="button"
                onClick={handleUseInTrain}
                className="mt-2 rounded-md bg-emerald-600 text-white px-3 py-1.5 text-sm hover:bg-emerald-700"
              >
                Use in Train →
              </button>
            </div>
          )}
        </aside>
      </section>
    </div>
  );
}

function SchemaSection({
  schema,
  columns,
  format,
  onChange,
  usingFile,
}: {
  schema: SchemaMapping;
  columns: string[];
  format: InputFormat;
  onChange: <K extends keyof SchemaMapping>(
    key: K,
    value: SchemaMapping[K],
  ) => void;
  usingFile: boolean;
}) {
  const targetable = format !== "jsonl";
  if (format === "jsonl") {
    return (
      <div className="space-y-2">
        <label className="block text-sm font-medium">Schema</label>
        <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3 space-y-2">
          <Toggle
            checked={schema.passthrough_chat}
            onChange={(v) => onChange("passthrough_chat", v)}
            label="Already chat-shaped (rows have a 'messages' key)"
            hint="Skip column mapping and just run cleaners on the existing rows."
          />
          {!schema.passthrough_chat && (
            <p className="text-xs text-zinc-500">
              JSONL rows that aren't already chat-shaped need column mapping —
              switch on passthrough above, or use CSV/TSV instead.
            </p>
          )}
        </div>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium">Schema mapping</label>
      <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3 space-y-3">
        <div className="flex items-center gap-2 text-sm">
          <span className="text-zinc-600">Target shape:</span>
          {(["chat", "completion"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => onChange("target", t)}
              className={`px-2.5 py-1 rounded ${
                schema.target === t
                  ? "bg-zinc-900 text-white"
                  : "bg-white border border-zinc-300"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        {schema.target === "chat" ? (
          <>
            <FieldPicker
              label="User field"
              value={schema.user_field ?? ""}
              columns={columns}
              onChange={(v) => onChange("user_field", v)}
              disabled={usingFile}
            />
            <FieldPicker
              label="Assistant field"
              value={schema.assistant_field ?? ""}
              columns={columns}
              onChange={(v) => onChange("assistant_field", v)}
              disabled={usingFile}
            />
          </>
        ) : (
          <>
            <FieldPicker
              label="Prompt field"
              value={schema.prompt_field ?? ""}
              columns={columns}
              onChange={(v) => onChange("prompt_field", v)}
              disabled={usingFile}
            />
            <FieldPicker
              label="Completion field"
              value={schema.completion_field ?? ""}
              columns={columns}
              onChange={(v) => onChange("completion_field", v)}
              disabled={usingFile}
            />
          </>
        )}
        {usingFile && targetable && (
          <p className="text-xs text-zinc-500">
            Field mapping is based on the file's header row at Build time.
          </p>
        )}
      </div>
    </div>
  );
}

function FieldPicker({
  label,
  value,
  columns,
  onChange,
  disabled,
}: {
  label: string;
  value: string;
  columns: string[];
  onChange: (v: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-1">
      <label className="block text-xs text-zinc-600">{label}</label>
      {columns.length > 0 ? (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="w-full rounded-md border border-zinc-300 px-2 py-1 text-sm disabled:bg-zinc-100"
        >
          <option value="">— pick —</option>
          {columns.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      ) : (
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          placeholder="column name"
          className="w-full rounded-md border border-zinc-300 px-2 py-1 text-sm disabled:bg-zinc-100"
        />
      )}
    </div>
  );
}

function Toggle({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5"
      />
      <span className="text-sm">
        {label}
        {hint && (
          <span className="block text-xs text-zinc-500 leading-relaxed">
            {hint}
          </span>
        )}
      </span>
    </label>
  );
}
