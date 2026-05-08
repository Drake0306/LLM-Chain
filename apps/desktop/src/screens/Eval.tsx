import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import type { Run } from "../api/client";
import { useApiClient } from "../api/hooks";

/**
 * Eval suite: side-by-side outputs from the base model vs the
 * trained adapter for a small set of prompts. Lets users see —
 * qualitatively — whether the fine-tune actually changed behavior
 * the way they wanted, beyond loss-curve numerology.
 *
 * Wire: POST /api/runs/{id}/eval streams ``status`` (load progress),
 * ``token`` ({role, prompt_index, text}) for each delta, ``done``
 * at the end. We render a table where rows are prompts and columns
 * are base / adapter; tokens append as they arrive.
 */
const DEFAULT_PROMPTS = [
  "Hello, who are you?",
  "Write a one-sentence summary of what you can help me with.",
  "Translate this to French: 'Good morning, how are you today?'",
];

interface EvalRow {
  prompt: string;
  base: string;
  adapter: string;
}

export function EvalScreen() {
  const api = useApiClient();
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<Run | null>(null);
  const [prompts, setPrompts] = useState<string>(DEFAULT_PROMPTS.join("\n"));
  const [maxTokens, setMaxTokens] = useState(128);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [rows, setRows] = useState<EvalRow[]>([]);
  const [errMsg, setErrMsg] = useState<string | null>(null);
  // Closer for the in-flight stream so the user can stop a long
  // eval (or unmount cleans up the fetch).
  const closerRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!api || !runId) return;
    api.getRun(runId).then(setRun).catch(() => undefined);
  }, [api, runId]);

  // Replace the generic placeholder set with whatever defaults the
  // sidecar has curated for this model's family — much more useful
  // out of the box than "Hello, who are you?". Only overrides when
  // the textarea is still showing the original DEFAULT_PROMPTS so
  // we don't clobber a user's custom edits.
  const sentinelDefault = DEFAULT_PROMPTS.join("\n");
  useEffect(() => {
    if (!api || !runId) return;
    api.getEvalDefaults(runId)
      .then(({ prompts: family }) => {
        if (family.length === 0) return;
        setPrompts((current) =>
          current === sentinelDefault ? family.join("\n") : current,
        );
      })
      .catch(() => undefined);
  }, [api, runId, sentinelDefault]);

  useEffect(() => {
    return () => {
      closerRef.current?.();
      closerRef.current = null;
    };
  }, []);

  function start() {
    if (!api || !runId || running) return;
    const list = prompts
      .split("\n")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    if (list.length === 0) {
      setErrMsg("Add at least one prompt — one per line.");
      return;
    }
    if (list.length > 20) {
      setErrMsg("Cap is 20 prompts per eval pass to keep runtime sane.");
      return;
    }
    setErrMsg(null);
    setStatus(null);
    setRunning(true);
    setRows(list.map((p) => ({ prompt: p, base: "", adapter: "" })));
    closerRef.current = api.evalRun(
      runId,
      { prompts: list, max_tokens: maxTokens },
      {
        onStatus: (msg) => setStatus(msg),
        onEval: (role, idx, text) => {
          setStatus(null);
          setRows((prev) => {
            if (idx < 0 || idx >= prev.length) return prev;
            const next = prev.slice();
            const cell = role === "base" ? "base" : "adapter";
            next[idx] = { ...next[idx], [cell]: next[idx][cell] + text };
            return next;
          });
        },
        onDone: () => {
          setRunning(false);
          setStatus(null);
          closerRef.current = null;
        },
        onError: (msg) => {
          setRunning(false);
          setStatus(null);
          setErrMsg(msg);
          closerRef.current = null;
        },
      },
    );
  }

  function stop() {
    closerRef.current?.();
    closerRef.current = null;
    setRunning(false);
    setStatus(null);
  }

  return (
    <div className="p-6 space-y-4 max-w-5xl">
      <header>
        <h1 className="text-2xl font-semibold">Eval</h1>
        {run && (
          <p className="text-sm text-zinc-500">
            <span className="font-medium">{run.config.model_id}</span> · run{" "}
            <Link to={`/runs/${run.id}`} className="font-mono text-xs underline">
              {run.id}
            </Link>
          </p>
        )}
      </header>

      <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 leading-relaxed">
        The base model loads first, then the adapter. Two model loads
        per eval pass — each can take 10–30 seconds depending on
        model size. Outputs use a low temperature (0.3) so before/after
        differences come from your training, not sampling noise.
      </p>

      <section className="space-y-3">
        <label className="text-sm space-y-1 block">
          <span className="block text-xs text-zinc-600">
            Prompts (one per line, max 20)
          </span>
          <textarea
            value={prompts}
            onChange={(e) => setPrompts(e.target.value)}
            disabled={running}
            rows={5}
            className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm font-mono"
          />
        </label>
        <div className="flex items-center gap-3">
          <label className="text-xs text-zinc-600 flex items-center gap-1.5">
            max tokens per prompt
            <input
              type="number"
              min={1}
              max={1024}
              value={maxTokens}
              disabled={running}
              onChange={(e) => {
                const next = parseInt(e.target.value, 10);
                if (Number.isFinite(next) && next > 0) setMaxTokens(next);
              }}
              className="w-20 rounded border border-zinc-300 px-2 py-1"
            />
          </label>
          <button
            type="button"
            onClick={start}
            disabled={running}
            className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
          >
            {running ? "Running…" : "Run eval"}
          </button>
          {running && (
            <button
              type="button"
              onClick={stop}
              className="rounded-md border border-red-200 text-red-700 px-4 py-2 text-sm hover:bg-red-50"
            >
              Stop
            </button>
          )}
          {running && (
            <button
              type="button"
              onClick={async () => {
                if (!api || !runId) return;
                try {
                  await api.skipEvalPrompt(runId);
                } catch (e) {
                  setErrMsg(String((e as Error).message ?? e));
                }
              }}
              title="Skip the prompt that's currently generating; continue with the rest."
              className="rounded-md border border-zinc-300 text-zinc-700 px-4 py-2 text-sm hover:bg-zinc-50"
            >
              Skip current
            </button>
          )}
          {status && (
            <span className="text-xs italic text-zinc-500">{status}</span>
          )}
        </div>
        {errMsg && (
          <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
            {errMsg}
          </div>
        )}
      </section>

      {rows.length > 0 && (
        <section className="border border-zinc-200 rounded-lg overflow-hidden">
          <div className="grid grid-cols-[1fr_1fr_1fr] gap-px bg-zinc-200">
            <div className="bg-zinc-50 px-3 py-2 text-xs uppercase tracking-wide text-zinc-500">
              Prompt
            </div>
            <div className="bg-zinc-50 px-3 py-2 text-xs uppercase tracking-wide text-zinc-500">
              Base model
            </div>
            <div className="bg-zinc-50 px-3 py-2 text-xs uppercase tracking-wide text-blue-700">
              Adapter (your fine-tune)
            </div>
            {rows.map((r, i) => (
              <div key={i} className="contents">
                <div className="bg-white px-3 py-3 text-sm whitespace-pre-wrap leading-relaxed">
                  {r.prompt}
                </div>
                <div className="bg-white px-3 py-3 text-sm whitespace-pre-wrap leading-relaxed text-zinc-700">
                  {r.base || (running ? "…" : "")}
                </div>
                <div className="bg-white px-3 py-3 text-sm whitespace-pre-wrap leading-relaxed">
                  {r.adapter || (running ? "…" : "")}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
