import type { DatasetChoice } from "./selection";

export type BackendPref = "auto" | "cuda" | "mlx";

export interface AppSettings {
  defaultBackend: BackendPref;
  defaultDatasetFormat: DatasetChoice["format"];
  defaultOutputDir: string;
  allowRestrictedModels: boolean;
}

const KEY = "llm-chain.settings.v1";

export const DEFAULT_SETTINGS: AppSettings = {
  defaultBackend: "auto",
  defaultDatasetFormat: "jsonl_chat",
  defaultOutputDir: "",
  allowRestrictedModels: false,
};

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_SETTINGS, ...parsed };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s: AppSettings): void {
  localStorage.setItem(KEY, JSON.stringify(s));
}
