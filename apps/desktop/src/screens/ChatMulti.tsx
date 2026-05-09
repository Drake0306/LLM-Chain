import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import type { MultiChatMessage, Run } from "../api/client";
import { useApiClient } from "../api/hooks";

interface AdapterColumn {
  runId: string;
  history: MultiChatMessage[];
}

const MAX_ADAPTERS = 3;

export function ChatMulti() {
  const api = useApiClient();
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [columns, setColumns] = useState<AdapterColumn[]>([]);
  const [input, setInput] = useState<string>("");
  const [maxTokens, setMaxTokens] = useState<number>(256);
  const [generating, setGenerating] = useState(false);
  const [statusByAdapter, setStatusByAdapter] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!api) return;
    api.listRuns().then((r) => {
      const succeeded = r.runs.filter(
        (x) =>
          x.status === "succeeded" &&
          x.config.backend !== "mlx_vlm" &&
          x.config.backend !== "cuda_vlm" &&
          // Hide LR-finder sweep runs from the picker — same rationale
          // as the comparator (sniff runs, not adapters worth chatting
          // with).
          x.config.purpose !== "lr_finder",
      );
      setRuns(succeeded);
    });
  }, [api]);

  useEffect(() => () => stopRef.current?.(), []);

  // Group runs by base model id so the picker can guide the user
  // toward valid combinations — multi-chat requires shared base.
  const runsByBase = useMemo(() => {
    const m = new Map<string, Run[]>();
    for (const r of runs) {
      const list = m.get(r.config.model_id) ?? [];
      list.push(r);
      m.set(r.config.model_id, list);
    }
    return m;
  }, [runs]);

  // Restrict the picker so a click on one run only enables others
  // sharing its base. Once at least one is selected, hide
  // incompatible options.
  const lockedBaseModelId = useMemo(() => {
    if (selected.length === 0) return null;
    const first = runs.find((r) => r.id === selected[0]);
    return first?.config.model_id ?? null;
  }, [selected, runs]);

  function toggleSelect(id: string) {
    if (generating) return;
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= MAX_ADAPTERS) return prev;
      return [...prev, id];
    });
  }

  function startSession() {
    setColumns(
      selected.map((rid) => ({
        runId: rid,
        history: [],
      })),
    );
    setStatusByAdapter({});
    setError(null);
  }

  function appendDeltaToLastAssistant(rid: string, delta: string) {
    setColumns((prev) =>
      prev.map((c) => {
        if (c.runId !== rid) return c;
        const last = c.history[c.history.length - 1];
        if (!last || last.role !== "assistant") {
          return {
            ...c,
            history: [...c.history, { role: "assistant", content: delta }],
          };
        }
        return {
          ...c,
          history: [
            ...c.history.slice(0, -1),
            { ...last, content: last.content + delta },
          ],
        };
      }),
    );
  }

  function send() {
    if (!api) return;
    const text = input.trim();
    if (!text || columns.length === 0) return;
    setError(null);

    // Append the new user turn to every adapter's history. We do this
    // *first* so the request body reflects the same history the UI
    // shows; the assistant turn is added empty per-adapter and then
    // grown as deltas arrive.
    setColumns((prev) =>
      prev.map((c) => ({
        ...c,
        history: [
          ...c.history,
          { role: "user", content: text },
          { role: "assistant", content: "" },
        ],
      })),
    );
    const generations = columns.map((c) => ({
      run_id: c.runId,
      messages: [
        ...c.history,
        { role: "user" as const, content: text },
      ],
    }));
    setInput("");
    setGenerating(true);
    stopRef.current = api.multiChat(
      { generations, max_tokens: maxTokens },
      {
        onToken: (adapterId, delta) =>
          appendDeltaToLastAssistant(adapterId, delta),
        onStatus: (adapterId, msg) =>
          setStatusByAdapter((s) => ({ ...s, [adapterId]: msg })),
        onDone: () => {
          setGenerating(false);
          setStatusByAdapter({});
        },
        onError: (msg) => {
          setError(msg);
          setGenerating(false);
          setStatusByAdapter({});
        },
      },
    );
  }

  function stop() {
    stopRef.current?.();
    stopRef.current = null;
    setGenerating(false);
    setStatusByAdapter({});
  }

  function reset() {
    if (generating) return;
    setColumns([]);
    setSelected([]);
    setInput("");
    setStatusByAdapter({});
    setError(null);
  }

  // Selection screen — shown until the user clicks Start session.
  if (columns.length === 0) {
    return (
      <div className="p-6 max-w-5xl space-y-6">
        <header className="flex items-baseline justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold">Multi-adapter chat</h1>
            <p className="text-sm text-zinc-500 leading-relaxed max-w-3xl">
              Pick 2–{MAX_ADAPTERS} succeeded runs that share a base model.
              Each one gets its own column and its own conversation
              history; you'll see how each adapter answers the same
              prompt side-by-side.
            </p>
          </div>
          <Link to="/library" className="text-sm text-zinc-600 hover:text-zinc-900">
            ← Back to Library
          </Link>
        </header>

        {runs.length < 2 && (
          <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2">
            Need at least 2 succeeded non-VLM runs in your library.
          </p>
        )}

        <div className="space-y-4">
          {Array.from(runsByBase.entries()).map(([modelId, group]) => {
            const groupValid = group.length >= 2;
            return (
              <div key={modelId} className="space-y-2">
                <h2 className="text-sm font-semibold text-zinc-700">
                  {modelId}
                  {!groupValid && (
                    <span className="ml-2 text-xs font-normal text-zinc-500">
                      need ≥ 2 runs on this base to compare
                    </span>
                  )}
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {group.map((r) => {
                    const checked = selected.includes(r.id);
                    const disabled =
                      !checked &&
                      ((lockedBaseModelId !== null &&
                        lockedBaseModelId !== r.config.model_id) ||
                        selected.length >= MAX_ADAPTERS);
                    return (
                      <label
                        key={r.id}
                        className={`flex items-center gap-2 px-3 py-2 rounded border ${
                          checked
                            ? "border-blue-300 bg-blue-50"
                            : "border-zinc-200 bg-white"
                        } ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={disabled}
                          onChange={() => toggleSelect(r.id)}
                        />
                        <span className="font-mono text-xs flex-1 truncate">
                          {r.id}
                        </span>
                        <span className="text-[10px] uppercase text-zinc-500">
                          {r.config.technique}
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex items-center gap-3 border-t border-zinc-200 pt-3">
          <button
            type="button"
            onClick={startSession}
            disabled={selected.length < 2}
            className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm disabled:bg-zinc-300"
          >
            Start session ({selected.length} selected)
          </button>
          <span className="text-xs text-zinc-500">
            Pick at least 2; max {MAX_ADAPTERS}.
          </span>
        </div>
      </div>
    );
  }

  // Chat session screen — N columns side-by-side.
  return (
    <div className="flex flex-col h-full">
      <header className="flex items-baseline justify-between gap-4 px-6 py-3 border-b border-zinc-200">
        <div>
          <h1 className="text-lg font-semibold">Multi-adapter chat</h1>
          <p className="text-xs text-zinc-500">
            {columns.length} adapters · base{" "}
            {lockedBaseModelId ?? "(unknown)"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={reset}
            disabled={generating}
            className="text-xs px-3 py-1.5 rounded-md border border-zinc-300 hover:bg-zinc-50 disabled:opacity-50"
          >
            Reset
          </button>
          <Link to="/library" className="text-xs text-zinc-600 hover:text-zinc-900">
            ← Library
          </Link>
        </div>
      </header>

      <div className="flex-1 overflow-hidden grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 divide-x divide-zinc-200">
        {columns.map((col) => (
          <ChatColumn
            key={col.runId}
            column={col}
            statusLine={statusByAdapter[col.runId]}
          />
        ))}
      </div>

      <footer className="border-t border-zinc-200 p-3 space-y-2">
        {error && (
          <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded p-2">
            {error}
          </div>
        )}
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              // Cmd/Ctrl+Enter sends — matches typical chat UX without
              // hijacking plain Enter for newlines in the textarea.
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                send();
              }
            }}
            disabled={generating}
            rows={2}
            placeholder="Ask all adapters something… (⌘/Ctrl+Enter to send)"
            className="flex-1 rounded-md border border-zinc-300 px-3 py-2 text-sm"
          />
          <div className="flex flex-col gap-1">
            <input
              type="number"
              value={maxTokens}
              min={16}
              max={4096}
              onChange={(e) =>
                setMaxTokens(Math.max(16, Math.min(4096, +e.target.value)))
              }
              disabled={generating}
              className="w-20 rounded-md border border-zinc-300 px-2 py-1 text-xs"
              title="max_tokens per adapter"
            />
            {generating ? (
              <button
                type="button"
                onClick={stop}
                className="rounded-md bg-red-600 text-white px-3 py-1.5 text-sm"
              >
                Stop
              </button>
            ) : (
              <button
                type="button"
                onClick={send}
                disabled={!input.trim()}
                className="rounded-md bg-blue-600 text-white px-3 py-1.5 text-sm disabled:bg-zinc-300"
              >
                Send
              </button>
            )}
          </div>
        </div>
      </footer>
    </div>
  );
}

function ChatColumn({
  column,
  statusLine,
}: {
  column: AdapterColumn;
  statusLine: string | undefined;
}) {
  return (
    <div className="flex flex-col min-w-0 overflow-hidden">
      <div className="px-3 py-2 border-b border-zinc-100 bg-zinc-50">
        <div className="font-mono text-xs text-zinc-500 truncate">
          {column.runId}
        </div>
        {statusLine && (
          <div className="text-[10px] text-blue-700 truncate">{statusLine}</div>
        )}
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {column.history.length === 0 ? (
          <div className="text-xs text-zinc-500">
            No messages yet. Send a prompt below.
          </div>
        ) : (
          column.history.map((msg, idx) => (
            <div
              key={idx}
              className={`text-sm leading-relaxed whitespace-pre-wrap break-words ${
                msg.role === "user" ? "text-zinc-900" : "text-zinc-700"
              }`}
            >
              <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-0.5">
                {msg.role}
              </div>
              <div>{msg.content || (msg.role === "assistant" ? "…" : "")}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
