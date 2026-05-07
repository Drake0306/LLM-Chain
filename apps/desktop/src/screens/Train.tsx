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

  async function startRun() {
    if (!api || !device || !model || !dataset) return;
    setBusy(true);
    setError(null);
    try {
      const cfg: RunConfig = {
        model_id: model.id,
        backend: device.backend,
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
    <div className="p-6 space-y-6 max-w-2xl">
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

      <section className="grid grid-cols-2 gap-4">
        <NumField label="Epochs" value={epochs} onChange={setEpochs} step={1} />
        <NumField label="Batch size" value={batchSize} onChange={setBatchSize} step={1} />
        <NumField label="Learning rate" value={lr} onChange={setLr} step={0.0001} />
        <NumField label="LoRA rank" value={loraRank} onChange={setLoraRank} step={1} />
        <NumField label="LoRA alpha" value={loraAlpha} onChange={setLoraAlpha} step={1} />
      </section>

      {error && (
        <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded p-2">
          {error}
        </div>
      )}

      <button
        type="button"
        onClick={startRun}
        disabled={!ready || busy}
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
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  step: number;
}) {
  return (
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
  );
}
