import { describe, expect, it } from "vitest";

import type { ModelEntry } from "../../api/client";
import {
  FORMAT_ORDER,
  FORMAT_SPECS,
  supportedFormats,
} from "../datasetSupport";

function makeModel(over: Partial<ModelEntry> = {}): ModelEntry {
  return {
    id: "test/model",
    name: "Test Model",
    family: "Test",
    params: 1_000_000_000,
    license: "Apache-2.0",
    license_caveat: null,
    modalities: ["text"],
    supports_lora: true,
    notes: null,
    restricted: false,
    chat_capable: true,
    ...over,
  };
}

describe("supportedFormats", () => {
  it("returns every format when no model is picked", () => {
    expect(supportedFormats(null)).toEqual(FORMAT_ORDER);
  });

  it("restricts vision-language models to chat-with-images only", () => {
    const vlm = makeModel({ modalities: ["text", "image"] });
    expect(supportedFormats(vlm)).toEqual(["jsonl_chat_vision"]);
  });

  it("hides chat formats from base text models without a chat template", () => {
    const base = makeModel({ chat_capable: false });
    const formats = supportedFormats(base);
    expect(formats).not.toContain("jsonl_chat");
    expect(formats).not.toContain("jsonl_chat_vision");
    // Plain-text formats are still allowed — the trainer trains on
    // {"text": ...} rows that don't need a chat template.
    expect(formats).toEqual(["csv", "text_dir", "hf_hub"]);
  });

  it("offers chat + plain-text formats to chat-tuned text models", () => {
    const chat = makeModel({ chat_capable: true });
    const formats = supportedFormats(chat);
    expect(formats).toContain("jsonl_chat");
    expect(formats).toContain("csv");
    expect(formats).toContain("text_dir");
    expect(formats).toContain("hf_hub");
    // Vision-only format must not leak onto a text model.
    expect(formats).not.toContain("jsonl_chat_vision");
  });
});

describe("FORMAT_SPECS", () => {
  it("has an entry for every format in FORMAT_ORDER", () => {
    for (const fmt of FORMAT_ORDER) {
      expect(FORMAT_SPECS[fmt]).toBeDefined();
      expect(FORMAT_SPECS[fmt].example.length).toBeGreaterThan(0);
      expect(FORMAT_SPECS[fmt].help.length).toBeGreaterThan(0);
    }
  });

  it("uses the same value as the keys for self-consistency", () => {
    for (const fmt of FORMAT_ORDER) {
      expect(FORMAT_SPECS[fmt].value).toBe(fmt);
    }
  });
});
