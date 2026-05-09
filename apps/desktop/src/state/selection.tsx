import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import type { HardwareDevice, ModelEntry } from "../api/client";

export type Technique = "lora" | "qlora";

export interface DatasetChoice {
  format:
    | "jsonl_chat"
    | "jsonl_chat_vision"
    | "csv"
    | "text_dir"
    | "hf_hub"
    | "jsonl_dpo";
  path?: string;
  hf_id?: string;
  text_column?: string;
}

interface SelectionState {
  device: HardwareDevice | null;
  model: ModelEntry | null;
  dataset: DatasetChoice | null;
  technique: Technique;
  setDevice: (d: HardwareDevice | null) => void;
  setModel: (m: ModelEntry | null) => void;
  setDataset: (d: DatasetChoice | null) => void;
  setTechnique: (t: Technique) => void;
}

const Ctx = createContext<SelectionState | null>(null);

// Versioned key so a future schema change can opt out of stale data
// without coordinating reads. Bumping this loses everyone's saved
// selection, but that's preferable to crashing on an incompatible
// shape.
const STORAGE_KEY = "llm-chain.selection.v1";

interface PersistedSelection {
  device: HardwareDevice | null;
  model: ModelEntry | null;
  dataset: DatasetChoice | null;
  technique: Technique;
}

function loadPersisted(): PersistedSelection {
  // Persist-then-reload behaviour: re-fetching the registry to
  // re-validate device/model on every mount would force us to gate
  // the whole UI behind a hardware probe. Instead we trust what we
  // saved — the per-screen gates (chat_capable, modalities, dataset
  // path existence, /api/runs validation) catch any inconsistency
  // when the user actually clicks Train. Worst case the user
  // re-picks once after an upgrade that retired their model.
  const empty: PersistedSelection = {
    device: null,
    model: null,
    dataset: null,
    technique: "qlora",
  };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return empty;
    const parsed = JSON.parse(raw) as Partial<PersistedSelection>;
    return {
      device: parsed.device ?? null,
      model: parsed.model ?? null,
      dataset: parsed.dataset ?? null,
      technique:
        parsed.technique === "lora" || parsed.technique === "qlora"
          ? parsed.technique
          : "qlora",
    };
  } catch {
    return empty;
  }
}

function persist(state: PersistedSelection): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // localStorage can fail in private mode or when quota is hit.
    // Silently no-op — selection works in-memory either way.
  }
}

export function SelectionProvider({ children }: { children: ReactNode }) {
  const initial = useMemo(() => loadPersisted(), []);
  const [device, setDevice] = useState<HardwareDevice | null>(initial.device);
  const [model, setModel] = useState<ModelEntry | null>(initial.model);
  const [dataset, setDataset] = useState<DatasetChoice | null>(initial.dataset);
  const [technique, setTechnique] = useState<Technique>(initial.technique);

  // Mirror state changes to localStorage so the next launch starts
  // where the user left off. Single useEffect covering all four
  // fields — JSON.stringify handles the diff at write time.
  useEffect(() => {
    persist({ device, model, dataset, technique });
  }, [device, model, dataset, technique]);

  const value = useMemo(
    () => ({ device, model, dataset, technique, setDevice, setModel, setDataset, setTechnique }),
    [device, model, dataset, technique],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSelection(): SelectionState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useSelection must be used inside <SelectionProvider>");
  return v;
}
