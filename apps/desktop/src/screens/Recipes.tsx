import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import type { CuratedDownloadState, Recipe } from "../api/client";
import { useApiClient } from "../api/hooks";
import { useSelection } from "../state/selection";

export function Recipes() {
  const api = useApiClient();
  const navigate = useNavigate();
  const { setModel, setDataset, setTechnique } = useSelection();

  const [recipes, setRecipes] = useState<Recipe[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [applying, setApplying] = useState<string | null>(null);
  const [statusLine, setStatusLine] = useState<string | null>(null);

  useEffect(() => {
    if (!api) return;
    let cancelled = false;
    api
      .listRecipes()
      .then((r) => !cancelled && setRecipes(r.recipes))
      .catch((e: unknown) =>
        !cancelled && setError(String((e as Error).message ?? e)),
      );
    return () => {
      cancelled = true;
    };
  }, [api]);

  async function applyRecipe(recipe: Recipe) {
    if (!api) return;
    if (recipe.needs_upgrade) return;
    setError(null);
    setApplying(recipe.id);
    setStatusLine("Applying recipe…");
    try {
      // 1. Resolve the model from the registry. Recipes ship a model id
      //    string; we look up the full ModelEntry so the downstream
      //    "supportedFormats" gating stays accurate.
      const modelsResp = await api.getModels(undefined, true);
      const modelEntry = modelsResp.models.find((m) => m.id === recipe.model);
      if (!modelEntry) {
        throw new Error(
          `Recipe references model ${recipe.model} which isn't in the ` +
            `registry. The recipe may have been authored against an older ` +
            `app version.`,
        );
      }
      setModel(modelEntry);
      setTechnique(recipe.technique);

      // 2. Resolve the dataset branch.
      if (recipe.dataset.kind === "curated" && recipe.dataset.curated_id) {
        const curatedId = recipe.dataset.curated_id;
        // If the dataset is already on disk (prior download), poll its
        // status to find the path; otherwise kick off the download
        // and wait for it to finish.
        setStatusLine(`Checking curated dataset ${curatedId}…`);
        const cached = await api.getCuratedDownloadStatus(curatedId);
        let finalState: CuratedDownloadState | null = cached;
        if (!cached || cached.status !== "done") {
          if (!cached || cached.status === "failed") {
            setStatusLine(
              `Downloading ${curatedId} from Hugging Face… (first ` +
                `time only; subsequent applies use the local cache)`,
            );
            await api.startCuratedDownload(curatedId);
          } else {
            setStatusLine(`Resuming ${curatedId} download…`);
          }
          // Poll until terminal.
          const deadline = Date.now() + 10 * 60 * 1000;
          while (Date.now() < deadline) {
            await new Promise((r) => setTimeout(r, 1000));
            const s = await api.getCuratedDownloadStatus(curatedId);
            if (!s) continue;
            finalState = s;
            if (s.status === "done" || s.status === "failed") break;
          }
        }
        if (!finalState || finalState.status !== "done" || !finalState.path) {
          throw new Error(
            finalState?.error ??
              "Curated dataset download didn't complete in time.",
          );
        }
        setDataset({ format: "jsonl_chat", path: finalState.path });
      } else if (recipe.dataset.kind === "synth") {
        // Synth recipes don't auto-generate — generation is lengthy
        // and the user should review the rows. Send them to the
        // synth screen with the topic+style pre-filled via query
        // params; that route reads them on mount.
        const params = new URLSearchParams({
          topic: recipe.dataset.synth_topic ?? "",
          style: recipe.dataset.synth_style ?? "",
        });
        setStatusLine("Synth recipe — taking you to the synth screen…");
        navigate(`/dataset/synth?${params.toString()}`);
        setApplying(null);
        return;
      } else {
        // bring_your_own — clear any stale dataset selection so the
        // user has to pick before clicking Start.
        setDataset(null);
      }

      // 3. Carry hyperparameters across via sessionStorage so the
      //    Train page can read them on mount. The selection context
      //    doesn't track HP; rather than thread state through every
      //    component, lean on sessionStorage as the side channel.
      sessionStorage.setItem(
        "llm-chain.recipe.hyperparameters",
        JSON.stringify(recipe.hyperparameters),
      );
      sessionStorage.setItem("llm-chain.recipe.applied", recipe.id);

      navigate("/train");
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setApplying(null);
      setStatusLine(null);
    }
  }

  return (
    <div className="p-6 max-w-5xl space-y-6">
      <header className="flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Recipes</h1>
          <p className="text-sm text-zinc-500 leading-relaxed max-w-3xl">
            One-click fine-tuning starters. Picks the model, technique,
            dataset, and hyperparameters; you confirm the device on the
            Dashboard and click Start.
          </p>
        </div>
        <Link
          to="/train"
          className="text-sm text-zinc-600 hover:text-zinc-900"
        >
          ← Back to Train
        </Link>
      </header>

      {error && (
        <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded p-2 leading-relaxed">
          {error}
        </div>
      )}
      {statusLine && (
        <div className="text-xs text-blue-800 bg-blue-50 border border-blue-200 rounded p-2 leading-relaxed">
          {statusLine}
        </div>
      )}

      {!recipes && !error && (
        <div className="text-sm text-zinc-500">Loading recipes…</div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {(recipes ?? []).map((recipe) => (
          <RecipeCard
            key={recipe.id}
            recipe={recipe}
            applying={applying === recipe.id}
            onApply={() => applyRecipe(recipe)}
          />
        ))}
      </div>
    </div>
  );
}

function RecipeCard({
  recipe,
  applying,
  onApply,
}: {
  recipe: Recipe;
  applying: boolean;
  onApply: () => void;
}) {
  const datasetLabel = (() => {
    switch (recipe.dataset.kind) {
      case "curated":
        return `curated · ${recipe.dataset.curated_id}`;
      case "synth":
        return "synth (you'll generate after)";
      case "bring_your_own":
        return "bring your own";
      default:
        return recipe.dataset.kind;
    }
  })();

  return (
    <div
      className={`rounded-lg border p-4 space-y-2 ${
        recipe.needs_upgrade
          ? "border-zinc-200 bg-zinc-50 opacity-70"
          : "border-zinc-200 bg-white"
      }`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-lg font-medium">{recipe.name}</h2>
        {recipe.needs_upgrade && (
          <span className="text-[10px] uppercase tracking-wide bg-amber-100 text-amber-900 px-2 py-0.5 rounded">
            needs newer app
          </span>
        )}
      </div>
      <p className="text-sm text-zinc-700 leading-relaxed">
        {recipe.description}
      </p>
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-800 font-mono">
          {recipe.model}
        </span>
        <span className="px-2 py-0.5 rounded bg-zinc-100 text-zinc-700 uppercase">
          {recipe.technique}
        </span>
        <span className="px-2 py-0.5 rounded bg-zinc-100 text-zinc-700">
          {datasetLabel}
        </span>
        {recipe.suggested_backend && (
          <span className="px-2 py-0.5 rounded bg-zinc-100 text-zinc-700">
            backend: {recipe.suggested_backend}
          </span>
        )}
      </div>
      <div className="text-[11px] text-zinc-500">
        epochs {recipe.hyperparameters.epochs} · lr{" "}
        {recipe.hyperparameters.learning_rate} · rank{" "}
        {recipe.hyperparameters.lora_rank} · α{" "}
        {recipe.hyperparameters.lora_alpha}
      </div>
      {recipe.notes && (
        <p className="text-xs text-zinc-600 leading-relaxed bg-amber-50 border border-amber-200 rounded p-2">
          {recipe.notes}
        </p>
      )}
      <div className="pt-2 border-t border-zinc-100">
        <button
          type="button"
          onClick={onApply}
          disabled={applying || recipe.needs_upgrade}
          className="rounded-md bg-blue-600 text-white px-3 py-1.5 text-sm disabled:bg-zinc-300"
        >
          {applying ? "Applying…" : "Use recipe"}
        </button>
      </div>
    </div>
  );
}
