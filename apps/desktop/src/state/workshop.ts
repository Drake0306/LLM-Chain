/**
 * Pure helpers for the dataset workshop screen.
 *
 * The actual write happens server-side via POST /api/datasets/build.
 * These functions only power the live preview — they let the user see
 * what their pasted data parses into before they hit Build, without
 * round-tripping every keystroke through the sidecar.
 *
 * Mirror the backend's parser/schema-detector closely enough that the
 * preview agrees with the eventual write. Intentional simplifications:
 *  - We tolerate parse errors (return what we got + an error message)
 *    so the preview doesn't disappear mid-edit. The Build endpoint is
 *    strict.
 *  - We don't round-trip through cleaners; the stats line in the UI
 *    is computed on the server response.
 */

export type InputFormat = "csv" | "tsv" | "jsonl";
export type TargetFormat = "chat" | "completion";

export interface SchemaMapping {
  target: TargetFormat;
  user_field?: string;
  assistant_field?: string;
  prompt_field?: string;
  completion_field?: string;
  passthrough_chat: boolean;
}

export interface ParseResult {
  rows: Record<string, unknown>[];
  /** Detected column names (CSV/TSV). Empty for jsonl. */
  columns: string[];
  /** Non-fatal parse error to surface inline; rows still contain
   * whatever parsed before the failure. */
  error: string | null;
}

const USER_HINTS = [
  "user",
  "question",
  "input",
  "prompt",
  "instruction",
  "human",
];
const ASSISTANT_HINTS = [
  "assistant",
  "answer",
  "output",
  "response",
  "completion",
  "ai",
  "bot",
];

/**
 * RFC-4180-ish CSV/TSV parser. Just enough to handle quoted fields
 * with embedded commas/newlines. Bigger than a one-liner because
 * paste-from-Excel routinely produces quoted multi-line cells, and
 * a naive split(",") would shred them across rows.
 */
function parseDelimited(
  text: string,
  delim: "," | "\t",
): { rows: string[][]; error: string | null } {
  const rows: string[][] = [];
  let cur: string[] = [];
  let field = "";
  let i = 0;
  let inQuotes = false;
  let error: string | null = null;
  while (i < text.length) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i += 1;
        continue;
      }
      field += ch;
      i += 1;
      continue;
    }
    if (ch === '"') {
      // Quotes only valid at the start of a field. Mid-field quotes
      // happen when users paste from Numbers / weird editors; treat
      // them as literal so the preview doesn't error out.
      if (field.length === 0) {
        inQuotes = true;
        i += 1;
        continue;
      }
      field += ch;
      i += 1;
      continue;
    }
    if (ch === delim) {
      cur.push(field);
      field = "";
      i += 1;
      continue;
    }
    if (ch === "\n" || ch === "\r") {
      cur.push(field);
      field = "";
      // Skip CRLF as a single newline.
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
      if (cur.length > 1 || cur[0] !== "") rows.push(cur);
      cur = [];
      i += 1;
      continue;
    }
    field += ch;
    i += 1;
  }
  if (inQuotes) {
    error = "Unclosed quote — check for a missing \" near the end.";
  }
  // Push the final field/row if the text didn't end with a newline.
  if (field.length > 0 || cur.length > 0) {
    cur.push(field);
    if (cur.length > 1 || cur[0] !== "") rows.push(cur);
  }
  return { rows, error };
}

export function parseText(text: string, fmt: InputFormat): ParseResult {
  if (!text || !text.trim()) {
    return { rows: [], columns: [], error: null };
  }
  if (fmt === "jsonl") {
    const out: Record<string, unknown>[] = [];
    let error: string | null = null;
    const lines = text.split("\n");
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i].trim();
      if (!line) continue;
      try {
        const obj = JSON.parse(line);
        if (obj && typeof obj === "object" && !Array.isArray(obj)) {
          out.push(obj as Record<string, unknown>);
        } else {
          error = `Row ${i + 1}: must be a JSON object.`;
          break;
        }
      } catch (e) {
        error = `Row ${i + 1}: ${(e as Error).message}`;
        break;
      }
    }
    return { rows: out, columns: [], error };
  }
  const delim = fmt === "csv" ? "," : "\t";
  const parsed = parseDelimited(text, delim);
  if (parsed.rows.length === 0) {
    return { rows: [], columns: [], error: parsed.error };
  }
  const [header, ...rest] = parsed.rows;
  const columns = header.map((h) => h.trim());
  const rows = rest.map((cells) => {
    const obj: Record<string, unknown> = {};
    columns.forEach((col, idx) => {
      obj[col] = cells[idx] ?? "";
    });
    return obj;
  });
  return { rows, columns, error: parsed.error };
}

export function detectSchema(rows: Record<string, unknown>[]): SchemaMapping {
  const sample = rows[0];
  if (!sample) return { target: "chat", passthrough_chat: false };
  if ("messages" in sample) {
    return { target: "chat", passthrough_chat: true };
  }
  const keysLower = new Map<string, string>();
  for (const k of Object.keys(sample)) keysLower.set(k.toLowerCase(), k);
  const user = USER_HINTS.find((h) => keysLower.has(h));
  const assistant = ASSISTANT_HINTS.find((h) => keysLower.has(h));
  if (user && assistant) {
    const userKey = keysLower.get(user)!;
    const assistantKey = keysLower.get(assistant)!;
    const promptLike = user === "prompt" || user === "instruction";
    const completionLike =
      assistant === "completion" || assistant === "response";
    if (promptLike && completionLike) {
      return {
        target: "completion",
        prompt_field: userKey,
        completion_field: assistantKey,
        passthrough_chat: false,
      };
    }
    return {
      target: "chat",
      user_field: userKey,
      assistant_field: assistantKey,
      passthrough_chat: false,
    };
  }
  return { target: "chat", passthrough_chat: false };
}

/** Format bytes as a small human label for the build-result toast. */
export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
