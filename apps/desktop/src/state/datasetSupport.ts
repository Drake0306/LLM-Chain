/**
 * Single source of truth for "which dataset formats can this model train on?"
 * and "what does a row of <format> look like?".
 *
 * The Models picker reads this to surface a bottom-hover summary when the
 * user picks a model; the Dataset picker reads it to gate the format
 * dropdown and show an inline example. Sharing the data here means the two
 * screens can never drift — change the supports table in one place.
 */
import type { ModelEntry } from "../api/client";
import type { DatasetChoice } from "./selection";

export type DatasetFormat = DatasetChoice["format"];

export interface FormatSpec {
  value: DatasetFormat;
  label: string;
  help: string;
  /**
   * Short label for the floating model summary (e.g. "JSONL chat").
   * Kept distinct from `label` so the picker rows can be more verbose.
   */
  shortLabel: string;
  /** Multi-line example shown in the dataset picker's preview pane. */
  example: string;
  /** What kind of file this format expects in the OS picker. */
  filePicker: "jsonl" | "csv" | "directory" | "none";
}

export const FORMAT_SPECS: Record<DatasetFormat, FormatSpec> = {
  jsonl_chat: {
    value: "jsonl_chat",
    label: "JSONL chat",
    shortLabel: "JSONL chat",
    help: 'One row per JSON object: {"messages":[{"role":"user","content":"…"},…]}',
    filePicker: "jsonl",
    example: `{"messages":[
  {"role":"user","content":"What's the capital of France?"},
  {"role":"assistant","content":"Paris."}
]}
{"messages":[
  {"role":"system","content":"You are a helpful assistant."},
  {"role":"user","content":"Translate 'hello' to Spanish."},
  {"role":"assistant","content":"Hola."}
]}`,
  },
  jsonl_chat_vision: {
    value: "jsonl_chat_vision",
    label: "JSONL chat with images",
    shortLabel: "JSONL chat + images",
    help:
      'OpenAI-style content arrays. {"messages":[{"role":"user","content":[' +
      '{"type":"image","path":"./img.png"},{"type":"text","text":"…"}]},…]}. ' +
      "Image paths are resolved relative to the JSONL file.",
    filePicker: "jsonl",
    example: `{"messages":[
  {"role":"user","content":[
    {"type":"image","path":"./cats/01.png"},
    {"type":"text","text":"What animal is this?"}
  ]},
  {"role":"assistant","content":"A tabby cat."}
]}
{"messages":[
  {"role":"user","content":[
    {"type":"image","path":"./diagrams/flow.png"},
    {"type":"text","text":"Describe this diagram."}
  ]},
  {"role":"assistant","content":"A four-stage pipeline: …"}
]}`,
  },
  csv: {
    value: "csv",
    label: "CSV",
    shortLabel: "CSV",
    help: "Pick a column that holds the text to train on.",
    filePicker: "csv",
    example: `text,source
"The quick brown fox jumps over the lazy dog.",pangrams
"To be, or not to be, that is the question.",shakespeare
"Lorem ipsum dolor sit amet, consectetur…",lorem`,
  },
  text_dir: {
    value: "text_dir",
    label: "Folder of .txt files",
    shortLabel: "Folder of .txt",
    help: "Each file becomes one row. Subdirectories are searched recursively.",
    filePicker: "directory",
    example: `corpus/
├── intro.txt        "Welcome to the lab notebook…"
├── chapters/
│   ├── 01.txt       "On the origin of species…"
│   └── 02.txt       "Variation under nature…"
└── notes/
    └── todo.txt     "Re-run the calibration…"`,
  },
  hf_hub: {
    value: "hf_hub",
    label: "Hugging Face Hub",
    shortLabel: "HF Hub",
    help: "Datasets ID like 'acme/my-dataset'.",
    filePicker: "none",
    example: `# Public datasets — no download step
HuggingFaceH4/no_robots
allenai/c4
databricks/databricks-dolly-15k

# For columns the loader doesn't auto-detect ('text', 'content',
# 'input'), set the Text column field on this page.`,
  },
  jsonl_dpo: {
    value: "jsonl_dpo",
    label: "JSONL DPO (preference pairs)",
    shortLabel: "JSONL DPO",
    help:
      'Each row has prompt / chosen / rejected. Used by DPO training only ' +
      '(switch the Training method on Train to "DPO"). HF backends only — ' +
      'mlx_lm has no DPO trainer yet.',
    filePicker: "jsonl",
    example: `{"prompt":"What's the capital of France?","chosen":"Paris.","rejected":"I'm not sure, maybe Berlin?"}
{"prompt":"Translate 'hello' to Spanish.","chosen":"Hola.","rejected":"Hello in Spanish."}
{"prompt":"Summarise: rain falls on roofs.","chosen":"Rain falls on roofs.","rejected":"Lots of words about rain and various other things."}`,
  },
};

/** Order shown in dropdowns / lists. Stable across screens. */
export const FORMAT_ORDER: DatasetFormat[] = [
  "jsonl_chat",
  "jsonl_chat_vision",
  "csv",
  "text_dir",
  "hf_hub",
  "jsonl_dpo",
];

/**
 * Which dataset formats a model can be trained on.
 *
 * - Vision-language models (image modality) only train on chat-with-images.
 * - Chat-capable text models train on chat or any plain-text format.
 * - Base text models (no chat template) skip chat entirely; they need
 *   a non-chat format because mlx_lm / HF apply_chat_template fails
 *   without a template.
 *
 * Returning the full list when no model is picked keeps the dataset
 * picker's UX intact when the user navigates straight to it before
 * choosing a model.
 */
export function supportedFormats(model: ModelEntry | null): DatasetFormat[] {
  if (!model) return FORMAT_ORDER.slice();
  if (model.modalities.includes("image")) {
    return ["jsonl_chat_vision"];
  }
  if (model.chat_capable) {
    return ["jsonl_chat", "csv", "text_dir", "hf_hub", "jsonl_dpo"];
  }
  // Base models without a chat template can still do DPO if they
  // support generation; the trainer doesn't apply a chat template
  // for DPO.
  return ["csv", "text_dir", "hf_hub", "jsonl_dpo"];
}

/** Format ↔ training method compatibility (F-C10).
 *
 * SFT runs on every existing format; DPO requires the dedicated
 * jsonl_dpo format because the loader/trainer expect prompt /
 * chosen / rejected fields. Returning the matrix as a function so
 * the Train page can gate the format dropdown to whatever the
 * currently-selected method allows.
 */
export function formatsForTrainingMethod(
  method: "sft" | "dpo",
): DatasetFormat[] {
  if (method === "dpo") return ["jsonl_dpo"];
  return ["jsonl_chat", "jsonl_chat_vision", "csv", "text_dir", "hf_hub"];
}
