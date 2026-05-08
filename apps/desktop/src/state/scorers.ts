/**
 * Lightweight per-row scorers for the A/B prompt comparator.
 *
 * The comparator runs the same prompt against two adapters and shows
 * the outputs side-by-side. These functions add quantitative columns
 * — length, sentiment, keyword presence — so the user can spot
 * differences without reading every paragraph.
 *
 * Deliberately NOT an LLM-as-judge — keeping this client-side means
 * no extra round-trips, no extra licensing concern, and no
 * "judge model agrees with itself" failure mode. The catch is that
 * these are heuristics; a higher score doesn't always mean "better",
 * just "different on this axis". We surface the winner per scorer
 * with a marker, not a sweeping verdict.
 */

export interface ScoreCell {
  /** The raw numeric score for this scorer on this output. */
  value: number;
  /** Pre-formatted label for the table cell. */
  label: string;
}

export interface ScoreSet {
  length: ScoreCell;
  sentiment: ScoreCell;
  keywords: ScoreCell;
}

export interface ScoringOptions {
  /** Lowercased keyword list. Empty means the keyword scorer reports 0
   * for both sides — UI hides the column when no keywords are set. */
  keywords: string[];
}

const POSITIVE_WORDS = [
  "good",
  "great",
  "happy",
  "love",
  "thanks",
  "please",
  "yes",
  "correct",
  "right",
  "easy",
  "helpful",
  "glad",
  "sure",
  "clear",
  "absolutely",
  "perfect",
];
// Intentionally omitted from the negative list:
// - "no" is the answer to half of yes/no questions, not a sentiment signal.
// - "sorry" doubles as both refusal ("I'm sorry, I can't…") and empathy
//   ("sorry to hear that"); biases safety-tuned models toward negative
//   regardless of substance.
// "refuse" / "unfortunately" stay in — narrower meaning, fewer false hits.
const NEGATIVE_WORDS = [
  "bad",
  "wrong",
  "hate",
  "fail",
  "issue",
  "problem",
  "broken",
  "difficult",
  "hard",
  "unfortunately",
  "cannot",
  "unable",
  "error",
  "denied",
  "refuse",
  "terrible",
];

const TOKEN_RE = /[a-z']+/g;

function tokenize(text: string): string[] {
  return Array.from(text.toLowerCase().matchAll(TOKEN_RE), (m) => m[0]);
}

/** Crude lexicon-based sentiment in [-1, 1]. Counts positive vs
 * negative tokens and normalises by total tokens. The scale isn't
 * calibrated to anything external; the comparator just needs the
 * relative ordering between left and right to be meaningful, which
 * a token-frequency lexicon delivers cheaply. */
export function sentimentScore(text: string): number {
  const tokens = tokenize(text);
  if (tokens.length === 0) return 0;
  const pos = tokens.filter((t) => POSITIVE_WORDS.includes(t)).length;
  const neg = tokens.filter((t) => NEGATIVE_WORDS.includes(t)).length;
  if (pos === 0 && neg === 0) return 0;
  return (pos - neg) / Math.max(tokens.length, 1);
}

/** Approximate token count using char/4 — same heuristic the trainer
 * uses for the workshop's length filter. Cheap and good enough for
 * rough length comparisons. */
export function approxTokens(text: string): number {
  return Math.round(text.length / 4);
}

/** Fraction of supplied keywords that appear at least once. Empty
 * keyword list returns 0 (and the UI hides the column). */
export function keywordScore(text: string, keywords: string[]): number {
  if (keywords.length === 0) return 0;
  const lower = text.toLowerCase();
  const hits = keywords.filter((k) => lower.includes(k.toLowerCase())).length;
  return hits / keywords.length;
}

export function scoreText(text: string, opts: ScoringOptions): ScoreSet {
  const len = approxTokens(text);
  const sent = sentimentScore(text);
  const kw = keywordScore(text, opts.keywords);
  return {
    length: {
      value: len,
      label: `${len} tok`,
    },
    sentiment: {
      value: sent,
      label: sent.toFixed(3),
    },
    keywords: {
      value: kw,
      label: opts.keywords.length > 0 ? `${Math.round(kw * 100)}%` : "—",
    },
  };
}

/** Returns "left" / "right" / "tie" given two scores and a direction.
 *
 * Direction matters: longer responses aren't always better, so the
 * comparator surfaces winner-per-scorer rather than a global verdict.
 * Length is reported neutrally — the UI shows it without a winner
 * marker so the user can interpret it themselves.
 */
export function pickWinner(
  left: number,
  right: number,
  direction: "higher" | "lower" | "neutral",
  epsilon = 1e-6,
): "left" | "right" | "tie" {
  if (direction === "neutral") return "tie";
  if (Math.abs(left - right) < epsilon) return "tie";
  if (direction === "higher") return left > right ? "left" : "right";
  return left < right ? "left" : "right";
}
