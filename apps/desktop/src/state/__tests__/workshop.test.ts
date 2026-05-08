import { describe, expect, it } from "vitest";

import { detectSchema, formatBytes, parseText } from "../workshop";

describe("workshop.parseText", () => {
  it("parses CSV into header-keyed dicts", () => {
    const r = parseText("user,assistant\nhi,hello\nbye,goodbye\n", "csv");
    expect(r.error).toBeNull();
    expect(r.columns).toEqual(["user", "assistant"]);
    expect(r.rows).toEqual([
      { user: "hi", assistant: "hello" },
      { user: "bye", assistant: "goodbye" },
    ]);
  });

  it("parses TSV with tab delimiter", () => {
    const r = parseText("user\tassistant\nhi\thello\n", "tsv");
    expect(r.rows).toEqual([{ user: "hi", assistant: "hello" }]);
  });

  it("handles quoted CSV cells with embedded commas and newlines", () => {
    const text =
      'q,a\n"Hello, world","line1\nline2"\n"escaped ""quote""","ok"\n';
    const r = parseText(text, "csv");
    expect(r.error).toBeNull();
    expect(r.rows).toEqual([
      { q: "Hello, world", a: "line1\nline2" },
      { q: 'escaped "quote"', a: "ok" },
    ]);
  });

  it("parses JSONL into row objects", () => {
    const text =
      '{"messages":[{"role":"user","content":"a"}]}\n' +
      '{"messages":[{"role":"user","content":"b"}]}\n';
    const r = parseText(text, "jsonl");
    expect(r.error).toBeNull();
    expect(r.rows.length).toBe(2);
  });

  it("returns the parse error inline without losing prior rows", () => {
    const r = parseText('{"messages":[]}\nthis is not json\n', "jsonl");
    expect(r.rows.length).toBe(1);
    expect(r.error).toMatch(/Row 2/);
  });

  it("returns empty for whitespace-only input", () => {
    expect(parseText("   \n\n", "csv").rows).toEqual([]);
    expect(parseText("", "jsonl").rows).toEqual([]);
  });
});

describe("workshop.detectSchema", () => {
  it("finds user/assistant columns", () => {
    const s = detectSchema([{ user: "a", assistant: "b" }]);
    expect(s).toMatchObject({
      target: "chat",
      user_field: "user",
      assistant_field: "assistant",
    });
  });

  it("recognises Question/Answer case-insensitively", () => {
    const s = detectSchema([{ Question: "?", Answer: "!" }]);
    expect(s).toMatchObject({
      user_field: "Question",
      assistant_field: "Answer",
    });
  });

  it("flags chat passthrough when rows already carry messages", () => {
    const s = detectSchema([
      { messages: [{ role: "user", content: "hi" }] },
    ]);
    expect(s.passthrough_chat).toBe(true);
  });

  it("picks completion target for prompt/completion columns", () => {
    const s = detectSchema([{ prompt: "p", completion: "c" }]);
    expect(s.target).toBe("completion");
  });

  it("falls back to chat with no fields when no hints match", () => {
    const s = detectSchema([{ foo: "x", bar: "y" }]);
    expect(s.target).toBe("chat");
    expect(s.user_field).toBeUndefined();
  });
});

describe("workshop.formatBytes", () => {
  it("renders with sensible units", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toMatch(/KB/);
    expect(formatBytes(5_242_880)).toMatch(/MB/);
  });
});
