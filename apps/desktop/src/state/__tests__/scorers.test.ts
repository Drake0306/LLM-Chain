import { describe, expect, it } from "vitest";

import {
  approxTokens,
  keywordScore,
  pickWinner,
  scoreText,
  sentimentScore,
} from "../scorers";

describe("approxTokens", () => {
  it("uses char/4 heuristic", () => {
    expect(approxTokens("12345678")).toBe(2);
    expect(approxTokens("")).toBe(0);
  });
});

describe("sentimentScore", () => {
  it("returns 0 for empty text", () => {
    expect(sentimentScore("")).toBe(0);
  });

  it("trends positive when positive tokens dominate", () => {
    const score = sentimentScore("great helpful thanks easy clear");
    expect(score).toBeGreaterThan(0);
  });

  it("trends negative when negative tokens dominate", () => {
    const score = sentimentScore("error denied cannot broken fail");
    expect(score).toBeLessThan(0);
  });

  it("returns 0 for neutral text with no lexicon hits", () => {
    expect(sentimentScore("the cat sat on the mat")).toBe(0);
  });
});

describe("keywordScore", () => {
  it("returns 0 with no keywords supplied", () => {
    expect(keywordScore("anything goes", [])).toBe(0);
  });

  it("returns hit fraction case-insensitively", () => {
    expect(keywordScore("Hello there friend", ["hello", "Friend"])).toBe(1);
    expect(keywordScore("only one keyword present", ["only", "missing"])).toBe(
      0.5,
    );
  });
});

describe("scoreText", () => {
  it("produces all three cells with formatted labels", () => {
    const s = scoreText("Hello, this is a great answer.", {
      keywords: ["great"],
    });
    expect(s.length.label).toMatch(/tok$/);
    expect(s.sentiment.value).toBeGreaterThan(0);
    expect(s.keywords.label).toBe("100%");
  });

  it("renders keyword cell as '—' when no keywords supplied", () => {
    const s = scoreText("anything", { keywords: [] });
    expect(s.keywords.label).toBe("—");
  });
});

describe("pickWinner", () => {
  it("respects direction higher", () => {
    expect(pickWinner(0.5, 0.2, "higher")).toBe("left");
    expect(pickWinner(0.2, 0.5, "higher")).toBe("right");
  });

  it("respects direction lower", () => {
    expect(pickWinner(2, 5, "lower")).toBe("left");
  });

  it("ties when both sides match within epsilon", () => {
    expect(pickWinner(0.5, 0.5, "higher")).toBe("tie");
  });

  it("returns tie for neutral direction regardless of values", () => {
    expect(pickWinner(1, 100, "neutral")).toBe("tie");
  });
});
