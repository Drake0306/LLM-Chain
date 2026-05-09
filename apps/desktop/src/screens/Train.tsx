import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

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
  // F-C10 / F-C11 training method. DPO requires jsonl_dpo + non-MLX
  // backend; distill requires a teacher_model_id + non-MLX backend.
  // Route layer enforces these but we surface them here so the user
  // gets a hint before submitting.
  const [trainingMethod, setTrainingMethod] = useState<"sft" | "dpo" | "distill">(
    "sft",
  );
  const [dpoBeta, setDpoBeta] = useState<number>(0.1);
  const [teacherModelId, setTeacherModelId] = useState<string>("");
  const [distillAlpha, setDistillAlpha] = useState<number>(0.5);
  const [distillTemperature, setDistillTemperature] = useState<number>(2.0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasetSize, setDatasetSize] = useState<number | null>(null);
  const [datasetSizeError, setDatasetSizeError] = useState<string | null>(null);
  // Distinguishes "still loading" (counted=false) from "loaded with no
  // result" (counted=true && datasetSize===null) so HF Hub datasets
  // without published split sizes don't get stuck on "counting…".
  const [datasetCounted, setDatasetCounted] = useState(false);
  const [showLrFinder, setShowLrFinder] = useState(false);
  const [showSchedule, setShowSchedule] = useState(false);
  // Default to "tonight, 02:00 local". Overnight is the dominant
  // scheduled-run use case — kicking off something that takes hours
  // while the user sleeps. Format is the value HTML datetime-local
  // inputs expect: "YYYY-MM-DDTHH:MM" with no timezone suffix.
  const defaultScheduleAt = (() => {
    const d = new Date();
    d.setDate(d.getDate() + 1);
    d.setHours(2, 0, 0, 0);
    const pad = (n: number) => String(n).padStart(2, "0");
    return (
      `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
      `T${pad(d.getHours())}:${pad(d.getMinutes())}`
    );
  })();
  const [scheduleAt, setScheduleAt] = useState<string>(defaultScheduleAt);
  const [scheduleFireIfMissed, setScheduleFireIfMissed] = useState(false);
  const [scheduleResult, setScheduleResult] = useState<string | null>(null);
  // Default sweep — log-spaced across two-and-a-half decades, which
  // catches the common "is my LR off by 10x?" mistake. The previous
  // heuristic [lr/2, lr, lr*2] only covered 4x range and missed
  // the dominant failure modes (LR too small to converge in 10
  // steps; LR too large and exploding). User can edit per call.
  const DEFAULT_LR_SWEEP = "1e-5,5e-5,1e-4,5e-4,1e-3";
  const [lrFinderSweep, setLrFinderSweep] = useState(DEFAULT_LR_SWEEP);
  const [lrFinderSteps, setLrFinderSteps] = useState(10);
  // Banner shown when the user navigated here via "Use recipe" — names
  // the recipe that applied so the user can confirm what was set.
  const [appliedRecipeId, setAppliedRecipeId] = useState<string | null>(null);

  // Pull recipe-applied hyperparameters from sessionStorage (Recipes
  // screen writes them when the user clicks "Use recipe"). Reading
  // here means the HP fields end up populated on first render so the
  // user can review before clicking Start.
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem("llm-chain.recipe.hyperparameters");
      const appliedId = sessionStorage.getItem("llm-chain.recipe.applied");
      if (!raw) return;
      const hp = JSON.parse(raw) as {
        epochs: number;
        learning_rate: number;
        lora_rank: number;
        lora_alpha: number;
        batch_size: number;
      };
      if (typeof hp.epochs === "number") setEpochs(hp.epochs);
      if (typeof hp.batch_size === "number") setBatchSize(hp.batch_size);
      if (typeof hp.learning_rate === "number") setLr(hp.learning_rate);
      if (typeof hp.lora_rank === "number") setLoraRank(hp.lora_rank);
      if (typeof hp.lora_alpha === "number") setLoraAlpha(hp.lora_alpha);
      if (appliedId) setAppliedRecipeId(appliedId);
      // Clear so a manual nav back to Train doesn't keep re-applying.
      sessionStorage.removeItem("llm-chain.recipe.hyperparameters");
      sessionStorage.removeItem("llm-chain.recipe.applied");
    } catch {
      // Quietly ignore — Train still works with its defaults.
    }
  }, []);

  const ready = api && device && model && dataset;

  // Probe the dataset size on mount / change via the dedicated count
  // endpoint, which doesn't parse rows (so it's fine on multi-GB
  // JSONLs). HF Hub goes through the dataset-card metadata and
  // returns either an exact row count (for cards that publish split
  // sizes) or null (which we render as "counted at training time").
  useEffect(() => {
    setDatasetSize(null);
    setDatasetSizeError(null);
    setDatasetCounted(false);
    if (!api || !dataset) return;
    const identifier =
      dataset.format === "hf_hub"
        ? dataset.hf_id?.trim()
        : dataset.path?.trim();
    if (!identifier) return;
    let cancelled = false;
    api
      .countDataset({
        dataset_path: identifier,
        dataset_format: dataset.format,
        text_column: dataset.format === "csv" ? dataset.text_column : undefined,
      })
      .then((res) => {
        if (cancelled) return;
        setDatasetSize(res.row_count);
        setDatasetCounted(true);
      })
      .catch((e) => {
        if (cancelled) return;
        setDatasetSizeError(String((e as Error).message ?? e));
        setDatasetCounted(true);
      });
    return () => {
      cancelled = true;
    };
  }, [api, dataset?.format, dataset?.path, dataset?.hf_id, dataset?.text_column]);

  // 90/10 split, with at least one row in valid — same math as the
  // sidecar's MlxSubprocessTrainer._stage_data, mirrored here so the
  // badge matches what actually happens at training time.
  function splitFor(rows: number): { train: number; valid: number } {
    const cut = Math.min(Math.max(1, Math.floor(rows * 0.9)), rows - 1);
    return { train: cut, valid: rows - cut };
  }

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
    // F-C10 DPO gating mirrors the sidecar — surface here so the
    // user fixes the mismatch before the round-trip.
    if (trainingMethod === "dpo") {
      if (dataset.format !== "jsonl_dpo") {
        return (
          "DPO training needs a 'JSONL DPO (preference pairs)' dataset " +
          "with prompt / chosen / rejected per row. Switch the dataset " +
          "format on the Dataset page or change the training method " +
          'back to "SFT".'
        );
      }
      if (device.backend === "mlx") {
        return (
          "DPO isn't supported on MLX yet — mlx_lm doesn't ship a DPO " +
          "trainer. Pick a CUDA / CPU / ROCm device on the Dashboard, " +
          'or change training method back to "SFT".'
        );
      }
    } else if (dataset.format === "jsonl_dpo") {
      return (
        "JSONL DPO datasets only train under the DPO training method. " +
        'Switch the Training method below to "DPO" or pick a different ' +
        "dataset format."
      );
    }
    if (trainingMethod === "distill") {
      if (!teacherModelId.trim()) {
        return (
          "Distillation needs a teacher model id (any HF id you have " +
          "access to). Pick something bigger than your student."
        );
      }
      if (teacherModelId.trim() === model.id) {
        return (
          "Teacher and student must differ — distilling a model into " +
          "itself is a no-op."
        );
      }
      if (device.backend === "mlx") {
        return (
          "Distillation isn't supported on MLX yet. Pick a CUDA / CPU " +
          'or ROCm device, or switch the training method back to "SFT".'
        );
      }
    }
    // Block before the round-trip when the dataset is known too small.
    // The sidecar would reject anyway (single-row → both splits overlap),
    // but catching it here saves a click and gives the user the same
    // actionable message before they're invested in a Start.
    if (datasetSize !== null && datasetSize < 2) {
      return (
        `Dataset has only ${datasetSize} row${datasetSize === 1 ? "" : "s"}. ` +
        "Training needs at least 2 so the train and validation splits don't " +
        "overlap. Add more rows to your dataset and try again."
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

  function buildConfig(): RunConfig | null {
    if (!device || !model || !dataset) return null;
    const cfg: RunConfig = {
      model_id: model.id,
      backend: resolveBackend(),
      technique,
      dataset_path: dataset.path ?? dataset.hf_id ?? "",
      dataset_format: dataset.format,
      text_column: dataset.text_column,
      epochs,
      batch_size: batchSize,
      learning_rate: lr,
      lora_rank: loraRank,
      lora_alpha: loraAlpha,
    };
    if (trainingMethod === "dpo") {
      cfg.training_method = "dpo";
      cfg.dpo_beta = dpoBeta;
    } else if (trainingMethod === "distill") {
      cfg.training_method = "distill";
      cfg.teacher_model_id = teacherModelId.trim();
      cfg.distill_alpha = distillAlpha;
      cfg.distill_temperature = distillTemperature;
    }
    return cfg;
  }

  async function startRun() {
    if (!api) return;
    const cfg = buildConfig();
    if (!cfg) return;
    if (preflightError()) {
      setError(preflightError());
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const { id } = await api.createRun(cfg);
      navigate(`/runs/${id}`);
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  async function scheduleRun() {
    if (!api) return;
    const cfg = buildConfig();
    if (!cfg) return;
    if (preflightError()) {
      setError(preflightError());
      return;
    }
    setBusy(true);
    setError(null);
    setScheduleResult(null);
    try {
      // datetime-local provides a wall-clock string with no timezone;
      // the Date constructor interprets it as local, then toISOString
      // converts to UTC for the wire. Mismatch between what the user
      // typed and what the server schedules is the bug to avoid here.
      const localDate = new Date(scheduleAt);
      if (Number.isNaN(localDate.getTime())) {
        setError("Pick a valid start time.");
        setBusy(false);
        return;
      }
      // DST detection: on a spring-forward day "02:30" doesn't exist.
      // Different browsers shift differently (Chromium → 03:30, older
      // engines NaN-adjacent). Round-trip through toISOString +
      // toLocaleString and compare against what the user actually
      // typed; warn if the wall-clock disagrees.
      const pad = (n: number) => String(n).padStart(2, "0");
      const echoed =
        `${localDate.getFullYear()}-${pad(localDate.getMonth() + 1)}-` +
        `${pad(localDate.getDate())}T${pad(localDate.getHours())}:` +
        `${pad(localDate.getMinutes())}`;
      if (echoed !== scheduleAt) {
        setError(
          `That wall-clock time doesn't exist locally (DST gap). The ` +
            `browser shifted it to ${localDate.toLocaleString()}. Pick ` +
            `another time or proceed by re-entering the shifted value.`,
        );
        setBusy(false);
        return;
      }
      const entry = await api.scheduleRun({
        config: cfg,
        start_at: localDate.toISOString(),
        fire_if_missed: scheduleFireIfMissed,
      });
      setShowSchedule(false);
      setScheduleResult(
        `Scheduled — fires at ${localDate.toLocaleString()} (id ${entry.id.slice(0, 8)})`,
      );
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  }

  async function startLrFinder() {
    if (!api) return;
    const cfg = buildConfig();
    if (!cfg) return;
    if (preflightError()) {
      setError(preflightError());
      return;
    }
    // Parse + validate the sweep string. Splitting on commas /
    // whitespace lets the user paste a typical hyperparameter list
    // ("1e-5, 5e-5, 1e-4") without thinking about format.
    const parsed = lrFinderSweep
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => parseFloat(s));
    if (parsed.length === 0 || parsed.some((x) => !Number.isFinite(x) || x <= 0)) {
      setError(
        "LR sweep must be a comma- or whitespace-separated list of positive numbers (e.g. 1e-5,5e-5,1e-4).",
      );
      return;
    }
    if (parsed.length > 6) {
      setError("LR sweep is capped at 6 entries per finder run.");
      return;
    }
    if (lrFinderSteps < 2 || lrFinderSteps > 200) {
      setError("Steps per run must be between 2 and 200.");
      return;
    }
    setShowLrFinder(false);
    setBusy(true);
    setError(null);
    try {
      const { run_ids } = await api.lrFinder({
        config: cfg,
        learning_rates: parsed,
        steps_per_run: lrFinderSteps,
      });
      navigate(`/runs/compare?ids=${run_ids.join(",")}&live=1`);
    } catch (e) {
      setError(String((e as Error).message ?? e));
      setBusy(false);
    }
  }

  return (
    <div className="p-6 space-y-6 max-w-3xl">
      <header className="flex items-baseline justify-between gap-4">
        <h1 className="text-2xl font-semibold">Train</h1>
        <Link
          to="/train/recipes"
          className="text-sm text-blue-700 hover:underline"
        >
          Browse recipes →
        </Link>
      </header>

      {appliedRecipeId && (
        <div className="text-xs text-emerald-900 bg-emerald-50 border border-emerald-200 rounded p-2 leading-relaxed">
          Applied recipe <strong>{appliedRecipeId}</strong>. Hyperparameters,
          model, technique, and dataset have been pre-filled — review below
          and click Start training.
        </div>
      )}

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
        <DatasetSplitBadge
          format={dataset?.format ?? null}
          rows={datasetSize}
          error={datasetSizeError}
          splitFor={splitFor}
          counted={datasetCounted}
        />
      </section>

      <section className="rounded-lg border border-zinc-200 p-4 space-y-3">
        <div>
          <h2 className="text-sm font-medium">Training method</h2>
          <p className="text-xs text-zinc-500">
            SFT (default) is supervised fine-tuning on chat / completion
            data. DPO trains on preference pairs (prompt / chosen /
            rejected) and is the standard recipe for "make this model
            less verbose / less hallucinatory" runs that pure SFT
            can't do.
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm">
          {(["sft", "dpo", "distill"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setTrainingMethod(m)}
              className={`px-3 py-1.5 rounded border ${
                trainingMethod === m
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white border-zinc-300 hover:bg-zinc-50"
              }`}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
        {trainingMethod === "dpo" && (
          <>
            <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 leading-relaxed">
              DPO requires the JSONL DPO dataset format and runs on
              CUDA / CPU / ROCm only — mlx_lm doesn't ship a DPO
              trainer yet.
            </p>
            <NumField
              label="DPO β"
              value={dpoBeta}
              onChange={setDpoBeta}
              step={0.01}
              help="KL-penalty weight. TRL's default is 0.1; lower values let the policy drift further from the reference model. Start with the default."
            />
          </>
        )}
        {trainingMethod === "distill" && (
          <>
            <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-2 leading-relaxed">
              Distillation loads the teacher (frozen) alongside the
              student (LoRA-wrapped) and trains the student against
              KL-div + CE. Tokenizers must match — pick a teacher
              from the same family as the student. CUDA / CPU /
              ROCm only.
            </p>
            <div className="space-y-2">
              <label className="block text-sm font-medium">
                Teacher model
                <span className="ml-2 text-xs font-normal text-zinc-500">
                  HF id, e.g. Qwen/Qwen3-7B
                </span>
              </label>
              <input
                value={teacherModelId}
                onChange={(e) => setTeacherModelId(e.target.value)}
                placeholder="Qwen/Qwen3-7B"
                className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm font-mono"
              />
            </div>
            <NumField
              label="Distill α"
              value={distillAlpha}
              onChange={setDistillAlpha}
              step={0.05}
              help="Weight on the standard CE loss. The KL term gets (1 - α). 0 = pure distillation, 1 = pure SFT, 0.5 = balanced (recommended start)."
            />
            <NumField
              label="Temperature"
              value={distillTemperature}
              onChange={setDistillTemperature}
              step={0.5}
              help="Softens both distributions before comparing. 2–4 is standard; higher temperatures flatten the targets and let the student mimic uncertain teacher outputs."
            />
          </>
        )}
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

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={startRun}
          disabled={!ready || busy || !!blocker}
          className="rounded-md bg-blue-600 text-white px-5 py-2 text-sm font-medium disabled:bg-zinc-300"
        >
          {busy ? "Starting…" : "Start training"}
        </button>
        <button
          type="button"
          onClick={() => {
            setError(null);
            setShowLrFinder(true);
          }}
          disabled={!ready || busy || !!blocker}
          title="Spawn short sniff runs at a sweep of learning rates and recommend the one with lowest loss."
          className="rounded-md border border-zinc-300 px-4 py-2 text-sm hover:bg-zinc-50 disabled:opacity-50"
        >
          Find best LR
        </button>
        <button
          type="button"
          onClick={() => {
            setError(null);
            setScheduleResult(null);
            setShowSchedule(true);
          }}
          disabled={!ready || busy || !!blocker}
          title="Persist this run to disk and fire its training at a chosen time."
          className="rounded-md border border-zinc-300 px-4 py-2 text-sm hover:bg-zinc-50 disabled:opacity-50"
        >
          Schedule…
        </button>
      </div>

      {scheduleResult && (
        <div className="text-xs text-emerald-800 bg-emerald-50 border border-emerald-200 rounded p-2">
          {scheduleResult} ·{" "}
          <span className="text-emerald-700">
            see Runs → Scheduled to manage.
          </span>
        </div>
      )}

      {showSchedule && (
        <section className="rounded-lg border border-blue-200 bg-blue-50/40 p-4 space-y-3">
          <header className="flex items-baseline justify-between">
            <h2 className="text-sm font-medium">Schedule this run</h2>
            <button
              type="button"
              onClick={() => setShowSchedule(false)}
              className="text-xs text-zinc-500 hover:text-zinc-700"
            >
              cancel
            </button>
          </header>
          <p className="text-xs text-zinc-700 leading-relaxed">
            The sidecar must be running when the timer fires — closing
            the desktop app or putting the laptop to sleep stops scheduled
            runs from kicking off. Best for "tonight at 02:00" overnight
            kickoffs.
          </p>
          <div className="space-y-2">
            <label className="block text-sm font-medium">Start at</label>
            <input
              type="datetime-local"
              value={scheduleAt}
              onChange={(e) => setScheduleAt(e.target.value)}
              className="rounded-md border border-zinc-300 px-3 py-2 text-sm"
            />
          </div>
          <label className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              checked={scheduleFireIfMissed}
              onChange={(e) => setScheduleFireIfMissed(e.target.checked)}
              className="mt-0.5"
            />
            <span>
              Fire on next sidecar startup if the time has already passed
              <span className="block text-xs text-zinc-500">
                Leave off for a strict "fire at this time, not before".
              </span>
            </span>
          </label>
          <div className="flex items-center gap-2 pt-1">
            <button
              type="button"
              onClick={scheduleRun}
              disabled={busy}
              className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm disabled:bg-zinc-300"
            >
              {busy ? "Scheduling…" : "Schedule"}
            </button>
          </div>
        </section>
      )}

      {showLrFinder && (
        <section className="rounded-lg border border-blue-200 bg-blue-50/40 p-4 space-y-3">
          <header className="flex items-baseline justify-between">
            <h2 className="text-sm font-medium">Find best learning rate</h2>
            <button
              type="button"
              onClick={() => setShowLrFinder(false)}
              className="text-xs text-zinc-500 hover:text-zinc-700"
            >
              cancel
            </button>
          </header>
          <p className="text-xs text-zinc-600 leading-relaxed">
            Spawns one short run per learning rate, runs them in
            sequence, and recommends the one whose loss is lowest at
            the step budget. Runs are tagged
            <code className="font-mono mx-1">purpose=lr_finder</code>
            and live alongside your normal runs.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-[1fr_140px] gap-3">
            <label className="text-sm space-y-1 block">
              <span className="block text-xs text-zinc-600">
                Learning rates (comma- or space-separated, max 6)
              </span>
              <input
                type="text"
                value={lrFinderSweep}
                onChange={(e) => setLrFinderSweep(e.target.value)}
                placeholder={DEFAULT_LR_SWEEP}
                className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm font-mono"
              />
            </label>
            <label className="text-sm space-y-1 block">
              <span className="block text-xs text-zinc-600">
                Steps per run
              </span>
              <input
                type="number"
                min={2}
                max={200}
                value={lrFinderSteps}
                onChange={(e) => {
                  const next = parseInt(e.target.value, 10);
                  if (Number.isFinite(next) && next >= 2 && next <= 200) {
                    setLrFinderSteps(next);
                  }
                }}
                className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
              />
            </label>
          </div>
          <button
            type="button"
            onClick={startLrFinder}
            disabled={busy}
            className="rounded-md bg-blue-600 text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
          >
            {busy ? "Starting…" : "Start sweep"}
          </button>
        </section>
      )}
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

/**
 * Pre-Start visibility into what training will actually see: total row
 * count plus the 90/10 train/valid split. Surfaces single-row datasets
 * (which the sidecar refuses to stage) so the user fixes the input
 * before clicking Start, not after.
 *
 * HF Hub is opted out — counting rows would force a download of the
 * entire dataset, which isn't appropriate as a side-effect of opening
 * the Train page. The sidecar still validates at run time.
 */
function DatasetSplitBadge({
  format,
  rows,
  error,
  splitFor,
  counted,
}: {
  format: string | null;
  rows: number | null;
  error: string | null;
  splitFor: (n: number) => { train: number; valid: number };
  /** True once a count attempt has resolved (success OR null result),
   * False while we're still waiting on the network. Lets us
   * distinguish "loading" from "loaded with no answer" so HF Hub
   * datasets that don't publish split sizes show the right hint. */
  counted: boolean;
}) {
  if (!format) return null;
  if (error) {
    return <Row label="Rows" value={`couldn't count: ${error}`} />;
  }
  if (!counted) {
    return <Row label="Rows" value="counting…" />;
  }
  if (rows === null) {
    // The count endpoint resolved but couldn't determine a number —
    // HF Hub dataset whose card metadata doesn't publish split sizes.
    return (
      <Row
        label="Rows"
        value="HF Hub — count unavailable, will be checked at training time"
      />
    );
  }
  if (rows < 2) {
    return (
      <Row
        label="Rows"
        value={`${rows} — too few for training (need ≥ 2)`}
      />
    );
  }
  const { train, valid } = splitFor(rows);
  return (
    <Row
      label="Rows"
      value={`${rows.toLocaleString()} → ${train.toLocaleString()} train / ${valid.toLocaleString()} valid`}
    />
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
          onChange={(e) => {
            // parseFloat("") and parseFloat(".") return NaN. The previous
            // implementation passed NaN through to the RunConfig, which the
            // sidecar now rejects with a 400 — but only after Start training
            // is clicked. Coerce to the previous valid value at the input
            // boundary so the form stays internally consistent.
            const next = parseFloat(e.target.value);
            onChange(Number.isFinite(next) ? next : value);
          }}
          className="w-full rounded-md border border-zinc-300 px-3 py-2 text-sm"
        />
      </label>
      {help && (
        <p className="text-xs text-zinc-600 leading-relaxed md:pt-7">{help}</p>
      )}
    </div>
  );
}
