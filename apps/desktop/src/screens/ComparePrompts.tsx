import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import type { Run } from "../api/client";
import { useApiClient } from "../api/hooks";
import { pickWinner, scoreText } from "../state/scorers";

interface CellState {
  text: string;
}

const DEFAULT_PROMPTS = [
  "Hello, who are you?",
  "Write a one-sentence summary of what you can help me with.",
  "Translate this to French: 'Good morning, how are you today?'",
];

export function ComparePrompts() {
  const api = useApiClient();
  const [runs, setRuns] = useState<Run[]>([]);
  const [leftId, setLeftId] = useState<string>("");
  const [rightId, setRightId] = useState<string>("");
  const [prompts, setPrompts] = useState<string[]>(DEFAULT_PROMPTS);
  const [maxTokens, setMaxTokens] = useState<number>(128);
  const [keywords, setKeywords] = useState<string>("");

  const [running, setRunning] = useState(false);
  const [statusLine, setStatusLine] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cells, setCells] = useState<{ left: CellState[]; right: CellState[] }>({
    left: [],
    right: [],
  });

  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!api) return;
    api.listRuns().then((r) => {
      const succeeded = r.runs.filter(
        (x) =>
          x.status === "succeeded" &&
          x.config.backend !== "mlx_vlm" &&
          x.config.backend !== "cuda_vlm",
      );
      setRuns(succeeded);
      if (succeeded.length > 0) {
        setLeftId((cur) => cur || succeeded[0].id);
        setRightId((cur) => cur || (succeeded[1]?.id ?? ""));
      }
    });
  }, [api]);

  useEffect(() => () => stopRef.current?.(), []);

  // Runs of two adapters with the same base model are the only
  // legal compare pair (server enforces this too). Show a warning if
  // the user has picked an incompatible pair so they can fix before
  // hitting Run.
  const leftRun = runs.find((r) => r.id === leftId);
  const rightRun = runs.find((r) => r.id === rightId);
  const baseMismatch =
    !!leftRun &&
    !!rightRun &&
    leftRun.id !== rightRun.id &&
    leftRun.config.model_id !== rightRun.config.model_id;

  const keywordList = useMemo(
    () =>
      keywords
        .split(/[,\s]+/)
        .map((k) => k.trim())
        .filter(Boolean),
    [keywords],
  );

  function start() {
    if (!api) return;
    if (!leftId || !rightId || leftId === rightId) {
      setError("Pick two different runs.");
      return;
    }
    if (baseMismatch) {
      setError("The two runs use different base models.");
      return;
    }
    setError(null);
    setStatusLine("Connecting…");
    setCells({
      left: prompts.map(() => ({ text: "" })),
      right: prompts.map(() => ({ text: "" })),
    });
    setRunning(true);
    stopRef.current = api.comparePrompts(
      {
        left_run_id: leftId,
        right_run_id: rightId,
        prompts,
        max_tokens: maxTokens,
      },
      {
        onToken: (role, idx, text) => {
          setCells((prev) => {
            const next = { ...prev, [role]: prev[role].slice() };
            const cur = next[role][idx] ?? { text: "" };
            next[role][idx] = { text: cur.text + text };
            return next;
          });
        },
        onStatus: (msg) => setStatusLine(msg),
        onDone: () => {
          setStatusLine(null);
          setRunning(false);
        },
        onError: (msg) => {
          setError(msg);
          setStatusLine(null);
          setRunning(false);
        },
      },
    );
  }

  function stop() {
    stopRef.current?.();
    stopRef.current = null;
    setRunning(false);
    setStatusLine(null);
  }

  async function skipCurrent() {
    if (!api) return;
    try {
      await api.skipComparePrompt(leftId, rightId);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  function setPromptText(value: string) {
    const list = value
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    setPrompts(list.length > 0 ? list : DEFAULT_PROMPTS);
  }

  return (
    <div className="p-6 max-w-7xl space-y-6">
      <header className="flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">A/B prompt comparator</h1>
          <p className="text-sm text-zinc-500 leading-relaxed max-w-3xl">
            Run the same prompts against two trained adapters from your
            library. Outputs are scored per row on length, sentiment,
            and keyword presence — winners are highlighted per scorer,
            not as a global verdict.
          </p>
        </div>
        <Link to="/runs" className="text-sm text-zinc-600 hover:text-zinc-900">
          ← Back to Runs
        </Link>
      </header>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <RunPicker
          label="Left run"
          value={leftId}
          runs={runs}
          onChange={setLeftId}
          disabled={running}
        />
        <RunPicker
          label="Right run"
          value={rightId}
          runs={runs}
          onChange={setRightId}
          disabled={running}
        />
      </section>

      {baseMismatch && (
        <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 leading-relaxed">
          The two runs use different base models. Compare needs a shared
          base so the same prompts hit equivalent tokenizers.
        </div>
      )}

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="space-y-2 lg:col-span-2">
          <label className="block text-sm font-medium">
            Prompts
            <span className="ml-2 text-xs font-normal text-zinc-500">
              one per line · {prompts.length} total
            </span>
          </label>
          <textarea
            value={prompts.join("\n")}
            onChange={(e) => setPromptText(e.target.value)}
            rows={6}
            disabled={running}
            className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm font-mono"
          />
        </div>
        <div className="space-y-3">
          <div className="space-y-1">
            <label className="block text-sm font-medium">Max tokens</label>
            <input
              type="number"
              value={maxTokens}
              min={16}
              max={1024}
              onChange={(e) =>
                setMaxTokens(Math.max(16, Math.min(1024, +e.target.value)))
              }
              disabled={running}
              className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
            />
          </div>
          <div className="space-y-1">
            <label className="block text-sm font-medium">
              Keywords
              <span className="ml-2 text-xs font-normal text-zinc-500">
                comma-separated
              </span>
            </label>
            <input
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              placeholder="polite, code, summary"
              disabled={running}
              className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
            />
          </div>
        </div>
      </section>

      <section className="flex items-center gap-3 border-t border-zinc-200 pt-3">
        {running ? (
          <>
            <button
              type="button"
              onClick={stop}
              className="rounded-md bg-red-600 text-white px-4 py-2 text-sm hover:bg-red-700"
            >
              Stop
            </button>
            <button
              type="button"
              onClick={skipCurrent}
              className="rounded-md bg-amber-500 text-white px-3 py-1.5 text-sm hover:bg-amber-600"
            >
              Skip current
            </button>
          </>
        ) : (
          <button
            type="button"
            onClick={start}
            disabled={!api || !leftId || !rightId || leftId === rightId || baseMismatch}
            className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm disabled:bg-zinc-300"
          >
            Run compare
          </button>
        )}
        {statusLine && <span className="text-xs text-zinc-600">{statusLine}</span>}
        {error && (
          <span className="text-xs text-red-700 leading-relaxed whitespace-pre-wrap">
            {error}
          </span>
        )}
      </section>

      <section className="space-y-3">
        {prompts.map((prompt, i) => (
          <ComparisonRow
            key={i}
            prompt={prompt}
            leftText={cells.left[i]?.text ?? ""}
            rightText={cells.right[i]?.text ?? ""}
            keywords={keywordList}
          />
        ))}
      </section>
    </div>
  );
}

function RunPicker({
  label,
  value,
  runs,
  onChange,
  disabled,
}: {
  label: string;
  value: string;
  runs: Run[];
  onChange: (id: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-1">
      <label className="block text-sm font-medium">{label}</label>
      {runs.length === 0 ? (
        <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2">
          No succeeded runs in your library yet.
        </p>
      ) : (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
        >
          <option value="">— pick —</option>
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id} — {r.config.model_id}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}

function ComparisonRow({
  prompt,
  leftText,
  rightText,
  keywords,
}: {
  prompt: string;
  leftText: string;
  rightText: string;
  keywords: string[];
}) {
  const opts = { keywords };
  const ls = scoreText(leftText, opts);
  const rs = scoreText(rightText, opts);
  // Length winner is intentionally neutral — longer isn't always
  // better, so the column shows both numbers without a winner badge.
  // Sentiment + keyword winners use higher-is-the-flagged-side; the
  // user can interpret what that means for their data.
  const sentimentWinner = pickWinner(ls.sentiment.value, rs.sentiment.value, "higher");
  const keywordWinner = pickWinner(ls.keywords.value, rs.keywords.value, "higher");
  const showKeywords = keywords.length > 0;

  return (
    <div className="rounded border border-zinc-200 bg-white p-3 space-y-2">
      <div className="text-xs font-mono text-zinc-500 truncate">{prompt}</div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <CellPanel
          text={leftText}
          scores={ls}
          sentimentWinner={sentimentWinner === "left"}
          keywordWinner={showKeywords && keywordWinner === "left"}
          showKeywords={showKeywords}
        />
        <CellPanel
          text={rightText}
          scores={rs}
          sentimentWinner={sentimentWinner === "right"}
          keywordWinner={showKeywords && keywordWinner === "right"}
          showKeywords={showKeywords}
        />
      </div>
    </div>
  );
}

function CellPanel({
  text,
  scores,
  sentimentWinner,
  keywordWinner,
  showKeywords,
}: {
  text: string;
  scores: ReturnType<typeof scoreText>;
  sentimentWinner: boolean;
  keywordWinner: boolean;
  showKeywords: boolean;
}) {
  return (
    <div className="space-y-2 min-w-0">
      <div className="text-xs whitespace-pre-wrap break-words leading-relaxed text-zinc-800 min-h-16 bg-zinc-50 rounded border border-zinc-200 p-2">
        {text || <span className="text-zinc-400">…</span>}
      </div>
      <div className="flex items-center gap-2 text-[10px] uppercase tracking-wide text-zinc-500">
        <span>{scores.length.label}</span>
        <span
          className={`px-1.5 py-0.5 rounded ${
            sentimentWinner
              ? "bg-emerald-100 text-emerald-800"
              : "bg-zinc-100 text-zinc-700"
          }`}
        >
          sent {scores.sentiment.label}
        </span>
        {showKeywords && (
          <span
            className={`px-1.5 py-0.5 rounded ${
              keywordWinner
                ? "bg-emerald-100 text-emerald-800"
                : "bg-zinc-100 text-zinc-700"
            }`}
          >
            kw {scores.keywords.label}
          </span>
        )}
      </div>
    </div>
  );
}
