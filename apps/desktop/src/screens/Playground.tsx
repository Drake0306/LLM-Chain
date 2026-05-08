import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import type { Run } from "../api/client";
import { useApiClient } from "../api/hooks";

interface Turn {
  /** "you" for the user's prompt, "model" for streamed output. */
  role: "you" | "model";
  text: string;
  /** While the model turn is still streaming we render a pulsing
   * indicator. Flips false on the SSE done event so the UI knows the
   * turn is final. */
  streaming?: boolean;
  /** Transient status from the sidecar — "Loading model into memory…"
   * or similar. Cleared as soon as the first real token arrives so
   * the line replaces itself with content rather than persisting. */
  status?: string | null;
  error?: string;
}

/**
 * Inference playground for a SUCCEEDED run. Loads the run's adapter on
 * top of its base model on first prompt; subsequent prompts reuse the
 * cached model in the sidecar. Streams tokens via SSE so the UI fills
 * the response progressively rather than waiting for completion.
 *
 * Runs as a thin chat-style UI — each prompt is independent (no chat
 * history is fed back to the model), which keeps the playground a
 * single-turn evaluator. A real multi-turn chat would need session
 * state on the sidecar; out of scope here.
 */
export function Playground() {
  const api = useApiClient();
  const { runId } = useParams<{ runId: string }>();
  const [run, setRun] = useState<Run | null>(null);
  const [prompt, setPrompt] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [streaming, setStreaming] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  // Closer for the active SSE connection so the user can stop a
  // long-running generation mid-stream. Cleared once the stream ends
  // naturally so the Stop button only shows during in-flight calls.
  const closerRef = useRef<(() => void) | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!api || !runId) return;
    api
      .getRun(runId)
      .then(setRun)
      .catch((e: unknown) => setLoadError(String((e as Error).message ?? e)));
  }, [api, runId]);

  // Keep the transcript scrolled to the latest turn while a stream is
  // in flight. Skipped for static loads so the user can scroll up to
  // re-read earlier outputs.
  useEffect(() => {
    if (streaming) {
      transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [turns, streaming]);

  // Disposing the SSE on unmount prevents a model.generate from
  // continuing after the user navigated away (the sidecar keeps the
  // Python loop running otherwise, holding the model in memory).
  useEffect(() => {
    return () => {
      closerRef.current?.();
      closerRef.current = null;
    };
  }, []);

  function send() {
    const text = prompt.trim();
    if (!api || !runId || !text || streaming) return;
    setStreaming(true);
    setLoadError(null);
    setPrompt("");
    setTurns((prev) => [
      ...prev,
      { role: "you", text },
      { role: "model", text: "", streaming: true },
    ]);
    closerRef.current = api.generateRun(
      runId,
      { prompt: text, max_tokens: maxTokens, temperature },
      {
        onToken: (chunk) =>
          setTurns((prev) => {
            const next = prev.slice();
            const last = next[next.length - 1];
            if (last && last.role === "model") {
              // First real token clears the status hint so the UI
              // shows generated content instead of "Loading model…".
              next[next.length - 1] = {
                ...last,
                text: last.text + chunk,
                status: null,
              };
            }
            return next;
          }),
        onStatus: (msg) =>
          setTurns((prev) => {
            const next = prev.slice();
            const last = next[next.length - 1];
            if (last && last.role === "model" && !last.text) {
              next[next.length - 1] = { ...last, status: msg };
            }
            return next;
          }),
        onDone: () => {
          setTurns((prev) => {
            const next = prev.slice();
            const last = next[next.length - 1];
            if (last && last.role === "model") {
              next[next.length - 1] = { ...last, streaming: false };
            }
            return next;
          });
          setStreaming(false);
          closerRef.current = null;
        },
        onError: (msg) => {
          setTurns((prev) => {
            const next = prev.slice();
            const last = next[next.length - 1];
            if (last && last.role === "model") {
              next[next.length - 1] = {
                ...last,
                streaming: false,
                error: msg,
              };
            }
            return next;
          });
          setStreaming(false);
          closerRef.current = null;
        },
      },
    );
  }

  function stop() {
    closerRef.current?.();
    closerRef.current = null;
    setStreaming(false);
    setTurns((prev) => {
      const next = prev.slice();
      const last = next[next.length - 1];
      if (last && last.role === "model") {
        next[next.length - 1] = { ...last, streaming: false };
      }
      return next;
    });
  }

  if (loadError) {
    return (
      <div className="p-6 space-y-2">
        <h1 className="text-2xl font-semibold">Playground</h1>
        <pre className="text-xs bg-red-50 border border-red-200 rounded p-3 whitespace-pre-wrap">
          {loadError}
        </pre>
      </div>
    );
  }
  if (!run) return <div className="p-6 text-zinc-500">Loading run…</div>;

  return (
    <div className="p-6 space-y-4 max-w-3xl">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Playground</h1>
          <p className="text-sm text-zinc-500">
            <span className="font-medium">{run.config.model_id}</span> + your
            adapter from run{" "}
            <Link to={`/runs/${run.id}`} className="font-mono text-xs underline">
              {run.id}
            </Link>
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <label className="flex items-center gap-1">
            max tokens
            <input
              type="number"
              min={1}
              max={4096}
              value={maxTokens}
              disabled={streaming}
              onChange={(e) => {
                const next = parseInt(e.target.value, 10);
                if (Number.isFinite(next) && next > 0) setMaxTokens(next);
              }}
              className="w-20 rounded border border-zinc-300 px-2 py-1 ml-1"
            />
          </label>
          <label className="flex items-center gap-1">
            temp
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              disabled={streaming}
              onChange={(e) => {
                const next = parseFloat(e.target.value);
                if (Number.isFinite(next) && next >= 0 && next <= 2) setTemperature(next);
              }}
              className="w-16 rounded border border-zinc-300 px-2 py-1 ml-1"
            />
          </label>
        </div>
      </header>

      <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 leading-relaxed">
        First prompt loads the adapter onto the base model and can take
        a few seconds; the model stays cached in the sidecar so follow-up
        prompts are instant. Each prompt is independent — the playground
        doesn't carry conversation history across turns.
      </p>

      <section className="border border-zinc-200 rounded-lg p-4 space-y-4 min-h-64 max-h-[60vh] overflow-y-auto">
        {turns.length === 0 ? (
          <div className="text-sm text-zinc-400 text-center py-12">
            Ask the model anything.
          </div>
        ) : (
          turns.map((t, i) => (
            <article
              key={i}
              className={`space-y-1 ${t.role === "you" ? "" : "pl-3 border-l-2 border-blue-200"}`}
            >
              <div className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                {t.role}
                {t.streaming && (
                  <span className="ml-2 inline-block w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
                )}
              </div>
              {t.status && !t.text && (
                <div className="text-xs italic text-zinc-500">
                  {t.status}
                </div>
              )}
              <div className="text-sm whitespace-pre-wrap leading-relaxed">
                {t.text || (t.streaming && !t.status ? "…" : "")}
              </div>
              {t.error && (
                <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded p-2">
                  {t.error}
                </div>
              )}
            </article>
          ))
        )}
        <div ref={transcriptEndRef} />
      </section>

      <div className="flex gap-2">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            // Cmd/Ctrl-Enter sends; plain Enter inserts a newline
            // (chat models often want multi-line prompts).
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Type a prompt… (⌘/Ctrl+Enter to send)"
          rows={3}
          className="flex-1 rounded-md border border-zinc-300 px-3 py-2 text-sm font-mono resize-y"
        />
        <div className="flex flex-col gap-2">
          <button
            type="button"
            onClick={send}
            disabled={streaming || !prompt.trim()}
            className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
          >
            Send
          </button>
          {streaming && (
            <button
              type="button"
              onClick={stop}
              className="rounded-md border border-red-200 text-red-700 px-4 py-2 text-sm hover:bg-red-50"
            >
              Stop
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
