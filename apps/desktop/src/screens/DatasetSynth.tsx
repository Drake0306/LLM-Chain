import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import type {
  ModelEntry,
  Run,
  SynthBody,
  SynthDoneStats,
  SynthRow,
} from "../api/client";
import { useApiClient } from "../api/hooks";
import { useSelection } from "../state/selection";

type SourceKind = "run" | "base";

export function DatasetSynth() {
  const api = useApiClient();
  const navigate = useNavigate();
  const { device, setDataset } = useSelection();
  // Recipes that ship a synth dataset hand off here with ?topic=&style=.
  // Reading the params on mount lets the user click Generate without
  // re-typing what the recipe author already specified.
  const [searchParams] = useSearchParams();

  const [runs, setRuns] = useState<Run[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);

  const [sourceKind, setSourceKind] = useState<SourceKind>("run");
  const [runId, setRunId] = useState<string>("");
  const [modelId, setModelId] = useState<string>("");
  const [backend, setBackend] = useState<string>(device?.backend ?? "cpu");

  const [topic, setTopic] = useState<string>(
    "A friendly customer-support assistant for an e-commerce store. " +
      "Each conversation should be a question about an order, return, " +
      "or shipping issue.",
  );
  const [style, setStyle] = useState<string>(
    "Polite, concise, asks for clarifying details before promising a fix.",
  );
  const [count, setCount] = useState<number>(10);
  const [maxTokens, setMaxTokens] = useState<number>(512);
  // Seed prompts let the user pin the topic of each generated row to
  // a specific angle ("billing question", "shipping delay", "lost
  // package"). Without seeds, every row gets the generic "vary your
  // question" instruction and convos bunch up around the same topic
  // even at temperature 0.9.
  const [seedPromptsText, setSeedPromptsText] = useState<string>("");

  const [generating, setGenerating] = useState(false);
  const [statusLine, setStatusLine] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<SynthRow[]>([]);
  const [doneStats, setDoneStats] = useState<SynthDoneStats | null>(null);
  const [name, setName] = useState<string>("");
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // Stop button needs the disposer the api returns. Hold it in a ref
  // so the cleanup effect can reach it without rerunning every state
  // change.
  const stopRef = useRef<(() => void) | null>(null);

  // Recipe handoff: pre-fill topic/style from URL params. Runs once
  // on mount; if the user later edits the textareas we don't want to
  // overwrite their changes when the URL re-renders.
  useEffect(() => {
    const t = searchParams.get("topic");
    const s = searchParams.get("style");
    if (t) setTopic(t);
    if (s) setStyle(s);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load picker data once on mount.
  useEffect(() => {
    if (!api) return;
    let cancelled = false;
    api.listRuns().then((r) => {
      if (cancelled) return;
      const succeeded = r.runs.filter(
        (x) =>
          x.status === "succeeded" &&
          // VLM runs don't support inference yet — surface the same
          // gate the synth backend will enforce so the user picks
          // something workable.
          x.config.backend !== "mlx_vlm" &&
          x.config.backend !== "cuda_vlm",
      );
      setRuns(succeeded);
      if (succeeded.length > 0) setRunId((cur) => cur || succeeded[0].id);
    });
    api.getModels(undefined, false, ["text"], true).then((r) => {
      if (cancelled) return;
      setModels(r.models);
      if (r.models.length > 0) setModelId((cur) => cur || r.models[0].id);
    });
    return () => {
      cancelled = true;
    };
  }, [api]);

  useEffect(() => {
    return () => {
      stopRef.current?.();
    };
  }, []);

  function start() {
    if (!api) return;
    setError(null);
    setRows([]);
    setDoneStats(null);
    setSavedPath(null);
    setStatusLine("Connecting…");
    setGenerating(true);

    const seedPrompts = seedPromptsText
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    const body: SynthBody = {
      topic: topic.trim(),
      style: style.trim(),
      count,
      max_tokens: maxTokens,
      seed_prompts: seedPrompts.length > 0 ? seedPrompts : undefined,
    };
    if (sourceKind === "run") {
      if (!runId) {
        setError("Pick a trained run.");
        setGenerating(false);
        return;
      }
      body.source_run_id = runId;
    } else {
      if (!modelId || !backend) {
        setError("Pick a model + backend.");
        setGenerating(false);
        return;
      }
      body.source_model_id = modelId;
      body.source_backend = backend;
    }

    stopRef.current = api.synthDataset(body, {
      onRow: (row) => {
        setRows((prev) => {
          // Rows can arrive out of order if a retry pushes one past
          // the next; keep the array indexed by `index`.
          const next = prev.slice();
          while (next.length <= row.index) {
            next.push({} as SynthRow);
          }
          next[row.index] = row;
          return next;
        });
      },
      onStatus: (msg) => setStatusLine(msg),
      onDone: (stats) => {
        setDoneStats(stats);
        setStatusLine(null);
        setGenerating(false);
      },
      onError: (msg) => {
        setError(msg);
        setStatusLine(null);
        setGenerating(false);
      },
    });
  }

  function stop() {
    stopRef.current?.();
    stopRef.current = null;
    setGenerating(false);
    setStatusLine(null);
  }

  async function save() {
    if (!api) return;
    const parsed = rows.filter((r) => r && r.parsed && r.messages);
    if (parsed.length < 2) {
      setError("Need at least 2 parseable rows to save.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const text = parsed
        .map((r) => JSON.stringify({ messages: r.messages }))
        .join("\n") + "\n";
      const result = await api.buildDataset({
        raw_text: text,
        input_format: "jsonl",
        passthrough_chat: true,
        // dedup on save so a retried row that produced an identical
        // conversation doesn't ship duplicates to the trainer.
        dedupe: true,
        role_balance: false,
        drop_empty: true,
        name: name.trim() || `synth-${Date.now()}`,
      });
      setSavedPath(result.path);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  function useInTrain() {
    if (!savedPath) return;
    setDataset({ format: "jsonl_chat", path: savedPath });
    navigate("/train");
  }

  const parsedCount = useMemo(
    () => rows.filter((r) => r && r.parsed).length,
    [rows],
  );
  const failedCount = useMemo(
    () => rows.filter((r) => r && !r.parsed && r.raw_text != null).length,
    [rows],
  );

  return (
    <div className="p-6 max-w-6xl space-y-6">
      <header className="flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Synthetic Data</h1>
          <p className="text-sm text-zinc-500 leading-relaxed max-w-3xl">
            Generate training conversations from an existing chat-capable
            run or a base model. The output is a normal JSONL chat file —
            review the rows, then save and train.
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
        <div className="space-y-5">
          <div className="space-y-2">
            <label className="block text-sm font-medium">Source</label>
            <div className="flex gap-2 text-sm">
              {(["run", "base"] as SourceKind[]).map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setSourceKind(k)}
                  disabled={generating}
                  className={`px-3 py-1.5 rounded-md border ${
                    sourceKind === k
                      ? "bg-blue-600 text-white border-blue-600"
                      : "bg-white border-zinc-300 hover:bg-zinc-50"
                  } disabled:opacity-50`}
                >
                  {k === "run" ? "Trained run" : "Base model"}
                </button>
              ))}
            </div>
            {sourceKind === "run" ? (
              runs.length === 0 ? (
                <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2">
                  No succeeded runs in your library. Train one or pick
                  "Base model" instead.
                </p>
              ) : (
                <select
                  value={runId}
                  onChange={(e) => setRunId(e.target.value)}
                  disabled={generating}
                  className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
                >
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.id} — {r.config.model_id}
                    </option>
                  ))}
                </select>
              )
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <select
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  disabled={generating}
                  className="rounded-md border border-zinc-300 px-3 py-2 text-sm"
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
                <select
                  value={backend}
                  onChange={(e) => setBackend(e.target.value)}
                  disabled={generating}
                  className="rounded-md border border-zinc-300 px-3 py-2 text-sm"
                >
                  {["mlx", "cuda", "cpu", "rocm"].map((b) => (
                    <option key={b} value={b}>
                      {b}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">Topic</label>
            <textarea
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              disabled={generating}
              rows={3}
              className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">Style</label>
            <textarea
              value={style}
              onChange={(e) => setStyle(e.target.value)}
              disabled={generating}
              rows={2}
              className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium">
              Seed prompts
              <span className="ml-2 text-xs font-normal text-zinc-500">
                optional · one per line · rotated across rows
              </span>
            </label>
            <textarea
              value={seedPromptsText}
              onChange={(e) => setSeedPromptsText(e.target.value)}
              disabled={generating}
              rows={3}
              placeholder={
                "billing question\nshipping delay\nrefund policy"
              }
              className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm font-mono"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <label className="block text-sm font-medium">
                Count
                <span className="ml-2 text-xs font-normal text-zinc-500">
                  rows to generate
                </span>
              </label>
              <input
                type="number"
                value={count}
                min={1}
                max={100}
                onChange={(e) =>
                  setCount(Math.max(1, Math.min(100, +e.target.value)))
                }
                disabled={generating}
                className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
              />
            </div>
            <div className="space-y-2">
              <label className="block text-sm font-medium">
                Max tokens
                <span className="ml-2 text-xs font-normal text-zinc-500">
                  per row
                </span>
              </label>
              <input
                type="number"
                value={maxTokens}
                min={64}
                max={4096}
                step={64}
                onChange={(e) =>
                  setMaxTokens(Math.max(64, Math.min(4096, +e.target.value)))
                }
                disabled={generating}
                className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
              />
            </div>
          </div>

          <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900 leading-relaxed">
            <strong className="font-semibold">License caveat:</strong>{" "}
            Synthetic data inherits the source model's licensing and
            biases. Check the source model's license before training a
            model you intend to distribute.
          </div>

          <div className="flex items-center gap-3 pt-2 border-t border-zinc-200">
            {generating ? (
              <button
                type="button"
                onClick={stop}
                className="rounded-md bg-red-600 text-white px-4 py-2 text-sm hover:bg-red-700"
              >
                Stop
              </button>
            ) : (
              <button
                type="button"
                onClick={start}
                disabled={!api}
                className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm disabled:bg-zinc-300"
              >
                Generate
              </button>
            )}
            {statusLine && (
              <span className="text-xs text-zinc-600">{statusLine}</span>
            )}
            {error && (
              <span className="text-xs text-red-700 leading-relaxed whitespace-pre-wrap">
                {error}
              </span>
            )}
          </div>
        </div>

        <aside className="space-y-3 min-w-0">
          <div className="flex items-baseline justify-between gap-3">
            <h2 className="text-sm font-semibold">
              Generated rows
              {rows.length > 0 && (
                <span className="ml-2 text-xs font-normal text-zinc-500">
                  {parsedCount} parsed, {failedCount} failed
                </span>
              )}
            </h2>
          </div>

          <div className="space-y-2 max-h-[28rem] overflow-y-auto pr-1">
            {rows.length === 0 ? (
              <p className="text-xs text-zinc-500">
                No rows yet. Click Generate to start streaming.
              </p>
            ) : (
              rows.map((row, i) =>
                row && row.raw_text != null ? (
                  <RowCard key={i} row={row} />
                ) : (
                  <div
                    key={i}
                    className="rounded border border-zinc-200 bg-zinc-50 p-2 text-xs text-zinc-500"
                  >
                    Row {i + 1} pending…
                  </div>
                ),
              )
            )}
          </div>

          {doneStats && (
            <div className="space-y-2 rounded-md border border-emerald-200 bg-emerald-50 p-3">
              <div className="text-sm font-medium text-emerald-900">
                Done — {doneStats.parsed_ok} of {doneStats.total} parsed
              </div>
              {doneStats.parse_failed > 0 && (
                <div className="text-xs text-emerald-900">
                  {doneStats.parse_failed} row
                  {doneStats.parse_failed === 1 ? "" : "s"} failed to
                  parse and won't be saved. Toggle a different model or
                  raise max_tokens if this is a recurring issue.
                </div>
              )}
              {savedPath ? (
                <div className="space-y-2">
                  <div className="text-xs font-mono break-all text-emerald-900">
                    {savedPath}
                  </div>
                  <button
                    type="button"
                    onClick={useInTrain}
                    className="rounded-md bg-emerald-600 text-white px-3 py-1.5 text-sm hover:bg-emerald-700"
                  >
                    Use in Train →
                  </button>
                </div>
              ) : (
                <>
                  {/* Re-state the license caveat at save time. The
                   * left-side banner is above the Generate button; by
                   * the time the user reaches Save they may have
                   * scrolled past it and won't see it again. */}
                  <div className="text-xs text-amber-900 bg-amber-100 border border-amber-300 rounded p-2 leading-relaxed">
                    Reminder: synthetic data inherits the source model's
                    license. Confirm before training a model you intend
                    to distribute.
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="dataset name"
                      className="flex-1 rounded-md border border-emerald-300 bg-white px-2 py-1 text-sm"
                    />
                    <button
                      type="button"
                      onClick={save}
                      disabled={saving || parsedCount < 2}
                      className="rounded-md bg-emerald-600 text-white px-3 py-1.5 text-sm hover:bg-emerald-700 disabled:bg-zinc-300"
                    >
                      {saving ? "Saving…" : "Save as JSONL"}
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </aside>
      </section>
    </div>
  );
}

function RowCard({ row }: { row: SynthRow }) {
  const [expand, setExpand] = useState(false);
  const ok = row.parsed && row.messages;
  return (
    <div
      className={`rounded border p-2 text-xs leading-relaxed ${
        ok
          ? "border-zinc-200 bg-white"
          : "border-red-200 bg-red-50"
      }`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-[10px] text-zinc-500">
          row {row.index + 1}
        </span>
        <button
          type="button"
          onClick={() => setExpand((v) => !v)}
          className="text-[10px] text-zinc-500 underline-offset-2 hover:underline"
        >
          {expand ? "hide raw" : "show raw"}
        </button>
      </div>
      {ok ? (
        <div className="space-y-1 mt-1">
          {row.messages!.map((m, i) => (
            <div key={i} className="space-y-0.5">
              <div className="text-[10px] font-mono uppercase text-zinc-500">
                {m.role}
              </div>
              <div className="whitespace-pre-wrap">{m.content}</div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-red-800 mt-1">Parse failed.</div>
      )}
      {expand && row.raw_text && (
        <pre className="mt-2 whitespace-pre-wrap text-[10px] bg-zinc-900 text-zinc-100 rounded p-2 overflow-x-auto">
          {row.raw_text}
        </pre>
      )}
    </div>
  );
}
