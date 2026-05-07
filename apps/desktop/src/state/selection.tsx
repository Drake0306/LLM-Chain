import { createContext, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";

import type { HardwareDevice, ModelEntry } from "../api/client";

export type Technique = "lora" | "qlora";

export interface DatasetChoice {
  format: "jsonl_chat" | "csv" | "text_dir" | "hf_hub";
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

export function SelectionProvider({ children }: { children: ReactNode }) {
  const [device, setDevice] = useState<HardwareDevice | null>(null);
  const [model, setModel] = useState<ModelEntry | null>(null);
  const [dataset, setDataset] = useState<DatasetChoice | null>(null);
  const [technique, setTechnique] = useState<Technique>("qlora");
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
