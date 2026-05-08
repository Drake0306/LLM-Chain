import { useEffect, useState } from "react";

import type { ModelEntry } from "../api/client";
import { useApiClient } from "../api/hooks";
import { FORMAT_SPECS, supportedFormats } from "../state/datasetSupport";
import { useSelection } from "../state/selection";
import { loadSettings } from "../state/settings";

function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  return n.toLocaleString();
}

export function ModelPicker() {
  const api = useApiClient();
  const { device, model, setModel, technique, setTechnique, dataset } = useSelection();
  const [models, setModels] = useState<ModelEntry[] | null>(null);

  // CPU has its own (much smaller) cap; QLoRA isn't supported there because
  // bnb 4-bit needs CUDA. We force LoRA on CPU and hand the picker the CPU
  // cap regardless of which technique button is highlighted.
  const isCpu = device?.backend === "cpu";
  useEffect(() => {
    if (isCpu && technique !== "lora") setTechnique("lora");
  }, [isCpu, technique, setTechnique]);
  const cap = isCpu
    ? device?.capabilities.cpu_max_params
    : technique === "qlora"
      ? device?.capabilities.qlora_max_params
      : device?.capabilities.lora_max_params;
  const includeRestricted = loadSettings().allowRestrictedModels;
  // Dataset format drives modality gating: a chat-vision dataset asks the
  // sidecar for VLMs only; any text format hides VLMs so the user doesn't
  // accidentally pair a multimodal base with text-only data.
  const isVisionDataset = dataset?.format === "jsonl_chat_vision";
  const isChatDataset =
    dataset?.format === "jsonl_chat" || dataset?.format === "jsonl_chat_vision";
  const requiredModalities = isVisionDataset ? ["text", "image"] : undefined;

  // If the user picked a model first and then a dataset that's incompatible
  // (e.g. Pythia base + JSONL chat), clear the selection so the picker can
  // re-gate. Without this, the model lingers across navigation and the
  // mismatch only surfaces at Train time.
  useEffect(() => {
    if (!model) return;
    if (isChatDataset && !model.chat_capable) {
      setModel(null);
      return;
    }
    if (isVisionDataset && !model.modalities.includes("image")) {
      setModel(null);
      return;
    }
    if (!isVisionDataset && model.modalities.includes("image")) {
      setModel(null);
    }
  }, [model, isChatDataset, isVisionDataset, setModel]);

  useEffect(() => {
    if (!api) return;
    api
      .getModels(cap, includeRestricted, requiredModalities, isChatDataset)
      .then((r) => {
        // Reverse-gate text datasets: hide multimodal entries unless the
        // dataset is explicitly vision. The sidecar can't distinguish
        // "no preference" from "text only" since modalities filter is
        // additive, so we strip on the client.
        const filtered = isVisionDataset
          ? r.models
          : r.models.filter((m) => !m.modalities.includes("image"));
        setModels(filtered);
      });
  }, [api, cap, includeRestricted, isVisionDataset, isChatDataset]);

  if (!device) {
    return (
      <div className="p-6 text-zinc-500">
        Pick a device on the Dashboard first.
      </div>
    );
  }

  return (
    // Bottom padding leaves room for the floating "supported datasets" bubble
    // so the last model card isn't ever hidden behind it on short viewports.
    <div className="p-6 pb-32 space-y-6 relative">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Pick a Model</h1>
          <p className="text-sm text-zinc-500">
            Showing models that fit {device.name} for {technique.toUpperCase()} (≤{" "}
            {cap ? formatParams(cap) : "?"}).
          </p>
        </div>
        <div className="flex gap-2 text-sm">
          {(isCpu ? (["lora"] as const) : (["qlora", "lora"] as const)).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTechnique(t)}
              className={`px-3 py-1 rounded-md border ${
                technique === t
                  ? "border-blue-500 bg-blue-50 text-blue-800"
                  : "border-zinc-200 text-zinc-600 hover:border-zinc-400"
              }`}
            >
              {t.toUpperCase()}
            </button>
          ))}
        </div>
      </header>

      {!isCpu && (
        <p className="text-xs text-zinc-600 bg-zinc-50 border border-zinc-200 rounded p-3 leading-relaxed">
          <span className="font-medium">QLoRA</span> quantizes the base model to
          4-bit so a 7B fits in 8 GB VRAM — pick this on consumer GPUs and
          Apple Silicon.{" "}
          <span className="font-medium">LoRA</span> keeps the base in full
          precision: slightly faster steps and slightly higher quality, but the
          size cap is lower because the base eats more VRAM.
          {isVisionDataset && (
            <>
              {" "}
              <span className="font-medium">Vision models</span> only show up
              while a chat-with-images dataset is selected.
            </>
          )}
        </p>
      )}

      {isChatDataset && (
        <p className="text-xs text-zinc-500">
          Showing only chat-capable models (with a tokenizer chat template).
          Base checkpoints like Pythia or Mistral-v0.3 are hidden — they need a
          plain-text dataset (CSV, folder of .txt, or HF Hub).
        </p>
      )}

      {!models && <div className="text-zinc-500">Loading…</div>}
      {models && models.length === 0 && (
        <div className="text-zinc-500">
          No models in the curated allowlist fit this device + technique. Try LoRA, or pick a
          smaller device tier.
        </div>
      )}

      <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {models?.map((m) => {
          const selected = model?.id === m.id;
          return (
            <li key={m.id}>
              <button
                type="button"
                onClick={() => setModel(m)}
                className={`w-full text-left rounded-lg border p-4 transition ${
                  selected
                    ? "border-blue-500 ring-2 ring-blue-200 bg-blue-50"
                    : "border-zinc-200 hover:border-zinc-400"
                }`}
              >
                <div className="flex items-baseline justify-between gap-2">
                  <div className="font-medium">{m.name}</div>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      m.restricted
                        ? "bg-amber-100 text-amber-800 border border-amber-300"
                        : "bg-zinc-100 text-zinc-700"
                    }`}
                  >
                    {m.license}
                  </span>
                </div>
                <div className="text-xs text-zinc-500 font-mono mt-1">{m.id}</div>
                <div className="text-xs text-zinc-600 mt-2">
                  {m.family} • {formatParams(m.params)} params
                </div>
                {m.restricted && m.license_caveat && (
                  <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 mt-2">
                    <span className="font-medium">License caveat:</span>{" "}
                    {m.license_caveat}
                  </div>
                )}
                {m.notes && (
                  <div className="text-xs text-zinc-500 mt-1">{m.notes}</div>
                )}
              </button>
            </li>
          );
        })}
      </ul>

      {model && <SupportedDatasetsBubble model={model} />}
    </div>
  );
}

/**
 * Floating summary that hovers near the bottom of the Models page once a
 * model is picked. Sticky so it stays visible while the user scrolls
 * through the model list, with a margin from the screen edge so it
 * reads as floating rather than docked. Negative horizontal margin
 * lets it visually span the section while the inner bubble keeps the
 * standard padding.
 */
function SupportedDatasetsBubble({ model }: { model: ModelEntry }) {
  const formats = supportedFormats(model);
  return (
    <div className="sticky bottom-4 z-10 -mx-6 px-6 pointer-events-none">
      <div className="pointer-events-auto rounded-xl border border-zinc-200 bg-white/95 backdrop-blur shadow-lg p-4">
        <div className="flex items-baseline justify-between gap-3">
          <div className="text-sm">
            <span className="font-medium">{model.name}</span>{" "}
            <span className="text-zinc-500">trains on these dataset formats</span>
          </div>
          <span className="text-xs text-zinc-400">
            head to <span className="font-medium text-zinc-600">Dataset</span> next
          </span>
        </div>
        <ul className="mt-3 flex flex-wrap gap-2">
          {formats.map((f) => {
            const spec = FORMAT_SPECS[f];
            return (
              <li
                key={f}
                title={spec.help}
                className="inline-flex items-center gap-1.5 rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-xs text-blue-900"
              >
                <span className="font-medium">{spec.shortLabel}</span>
              </li>
            );
          })}
        </ul>
        <p className="mt-3 text-xs text-zinc-500 leading-relaxed">
          {model.modalities.includes("image")
            ? "Vision-language model — pair with chat-with-images JSONL only."
            : model.chat_capable
              ? "Chat-tuned — accepts JSONL chat or any plain-text format."
              : "Base checkpoint (no chat template) — use a plain-text format."}
        </p>
      </div>
    </div>
  );
}
