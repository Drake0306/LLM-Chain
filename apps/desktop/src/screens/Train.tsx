import { useState } from "react";
import { useNavigate } from "react-router-dom";

import type { RunConfig } from "../api/client";
import { useApiClient } from "../api/hooks";
import { useSelection } from "../state/selection";

export function Train() {
  const api = useApiClient();
  const navigate = useNavigate();
  const { device, model, dataset, technique } = useSelection();

  const [epochs, setEpochs] = useState(1);
  const [batchSize, setBatchSize] = useState(1);
  const [lr, setLr] = useState(2e-4);
  const [loraRank, setLoraRank] = useState(16);
  const [loraAlpha, setLoraAlpha] = useState(32);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ready = api && device && model && dataset;

  // Pre-flight: catch incompatibilities before we POST. The sidecar
  // validates too (defense in depth), but this gives instant feedback so
  // the user doesn't see a confusing round-trip + traceback.
  function preflightError(): string | null {
    if (!device || !model || !dataset) return null;
    const isChat =
      dataset.format === "jsonl_chat" || dataset.format === "jsonl_chat_vision";
    if (isChat && !model.chat_capable) {
      return (
        `${model.name} is a base model with no chat template — the ` +
        `"${dataset.format}" dataset format won't work on it. Pick a ` +
        "chat-capable model on the Models page (Qwen3-0.6B, SmolLM2 360M " +
        "Instruct, TinyLlama 1.1B Chat, etc.), or change the dataset to " +
        "CSV / text-dir / HF Hub."
      );
    }
    if (dataset.format === "jsonl_chat_vision" && !model.modalities.includes("image")) {
      return (
        `${model.name} is text-only — pair it with a JSONL chat (text) dataset, ` +
        "or pick a vision-language model like Qwen2-VL."
      );
    }
    return null;
  }
  const blocker = preflightError();

  // Vision datasets train on the VLM backends; text datasets stay on the
  // existing cuda/cpu/mlx paths. We pick the right trainer here so the user
  // never has to think about it.
  function resolveBackend(): string {
    if (!device) return "cuda";
    if (dataset?.format === "jsonl_chat_vision") {
      if (device.backend === "mlx") return "mlx_vlm";
      return "cuda_vlm";
    }
    return device.backend;
  }

  async function startRun() {
    if (!api || !device || !model || !dataset) return;
    if (preflightError()) {
      setError(preflightError());
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const cfg: RunConfig = {
        model_id: model.id,
        backend: resolveBackend(),
        technique,
        dataset_path: dataset.path ?? dataset.hf_id ?? "",
        dataset_format: dataset.format,
        epochs,
        batch_size: batchSize,
        learning_rate: lr,
        lora_rank: loraRank,
        lora_alpha: loraAlpha,
      };
      const { id } = await api.createRun(cfg);
      navigate(`/runs/${id}`);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  return (
    <div className="p-6 space-y-6 max-w-3xl">
      <header>
        <h1 className="text-2xl font-semibold">Train</h1>
      </header>

      <section className="rounded-lg border border-zinc-200 p-4 space-y-2 text-sm">
        <Row label="Device" value={device ? `${device.name} (${device.backend})` : "— pick on Dashboard"} />
        <Row label="Model" value={model ? `${model.name} (${model.id})` : "— pick on Models"} />
        <Row
          label="Dataset"
          value={
            dataset
              ? `${dataset.format} → ${dataset.path ?? dataset.hf_id}`
              : "— pick on Dataset"
          }
        />
        <Row label="Technique" value={technique.toUpperCase()} />
      </section>

      <section className="space-y-4">
        <NumField
          label="Epochs"
          value={epochs}
          onChange={setEpochs}
          step={1}
          help="One epoch = one full pass through your dataset. For LoRA on small datasets, 1 is usually enough; bump to 2–3 only if the loss curve is still trending down at the end of the first pass."
        />
        <NumField
          label="Batch size"
          value={batchSize}
          onChange={setBatchSize}
          step={1}
          help="How many examples the GPU processes at once. Bigger = faster and more stable training, but uses more VRAM. Start at 1 and raise only if your device has headroom and you're seeing CUDA / MLX OOM-free."
        />
        <NumField
          label="Learning rate"
          value={lr}
          onChange={setLr}
          step={0.0001}
          help="How aggressively the adapter updates each step. 2e-4 (0.0002) is the standard LoRA default and works for most cases. Halve it if loss spikes or goes NaN; double it if loss is plateauing too high."
        />
        <NumField
          label="LoRA rank"
          value={loraRank}
          onChange={setLoraRank}
          step={1}
          help="Size of the low-rank adapter. Higher rank = more capacity to learn, but a bigger adapter file and more VRAM. Rank 8–16 covers most chat / instruction fine-tunes. Go to 32–64 only for harder tasks (style transfer, code, multilingual)."
        />
        <NumField
          label="LoRA alpha"
          value={loraAlpha}
          onChange={setLoraAlpha}
          step={1}
          help="Scales how much the adapter influences the base model's output. Convention: alpha = 2 × rank (so 32 if rank is 16). Rarely needs to be tweaked separately — change rank and let alpha follow."
        />
      </section>

      {blocker && (
        <div className="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded p-3 leading-relaxed">
          {blocker}
        </div>
      )}

      {error && !blocker && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
          {error}
        </div>
      )}

      <button
        type="button"
        onClick={startRun}
        disabled={!ready || busy || !!blocker}
        className="rounded-md bg-blue-600 text-white px-5 py-2 text-sm font-medium disabled:bg-zinc-300"
      >
        {busy ? "Starting…" : "Start training"}
      </button>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-zinc-500">{label}</dt>
      <dd className="text-zinc-900 text-right">{value}</dd>
    </div>
  );
}

function NumField({
  label,
  value,
  onChange,
  step,
  help,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  step: number;
  help?: string;
}) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-4 items-start">
      <label className="space-y-1">
        <span className="block text-sm font-medium">{label}</span>
        <input
          type="number"
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
        />
      </label>
      {help && (
        <p className="text-xs text-zinc-600 leading-relaxed md:pt-7">{help}</p>
      )}
    </div>
  );
}
