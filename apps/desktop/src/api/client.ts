export type MemoryKind = "dedicated" | "unified" | "shared";

export interface DeviceCapabilities {
  qlora_max_params: number;
  lora_max_params: number;
  full_ft_max_params: number;
  notes: string;
  warning_codes: string[];
}

export interface HardwareDevice {
  backend: string;
  name: string;
  vram_gb: number;
  memory_kind: MemoryKind;
  driver_version: string | null;
  capabilities: DeviceCapabilities;
}

export interface HardwareReport {
  os: string;
  os_version: string;
  cpu: { cores: number; name: string };
  system_ram_gb: number;
  devices: HardwareDevice[];
}

export interface ModelEntry {
  id: string;
  name: string;
  family: string;
  params: number;
  license: string;
  license_caveat: string | null;
  modalities: string[];
  supports_lora: boolean;
  notes: string | null;
}

export interface RunConfig {
  model_id: string;
  backend: string;
  technique: "lora" | "qlora";
  dataset_path: string;
  dataset_format?: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  lora_rank?: number;
  lora_alpha?: number;
}

export interface Run {
  id: string;
  created_at: string;
  status: "pending" | "running" | "succeeded" | "failed" | "canceled";
  config: RunConfig;
  error: string | null;
  output_dir: string | null;
}

export interface TrainingEventPayload {
  type: "start" | "step" | "epoch_end" | "done" | "error";
  step: number;
  total_steps: number;
  epoch: number;
  loss: number | null;
  lr: number | null;
  message: string | null;
}

export class ApiClient {
  constructor(private port: number, private fetchImpl: typeof fetch = fetch) {}

  private base(path: string) {
    return `http://127.0.0.1:${this.port}${path}`;
  }

  async getHardware(): Promise<HardwareReport> {
    const r = await this.fetchImpl(this.base("/api/hardware"));
    return r.json();
  }

  async getModels(maxParams?: number): Promise<{ models: ModelEntry[] }> {
    const q = maxParams ? `?max_params=${maxParams}` : "";
    const r = await this.fetchImpl(this.base(`/api/models${q}`));
    return r.json();
  }

  async createRun(cfg: RunConfig): Promise<{ id: string; status: string }> {
    const r = await this.fetchImpl(this.base("/api/runs"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    return r.json();
  }

  async listRuns(): Promise<{ runs: Run[] }> {
    const r = await this.fetchImpl(this.base("/api/runs"));
    return r.json();
  }

  async getRun(runId: string): Promise<Run> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}`));
    return r.json();
  }

  streamRun(
    runId: string,
    onEvent: (ev: { type: string; payload: TrainingEventPayload }) => void,
  ): () => void {
    const es = new EventSource(this.base(`/api/runs/${runId}/stream`));
    const types = ["start", "step", "epoch_end", "done", "error"] as const;
    types.forEach((t) => {
      es.addEventListener(t, (e: MessageEvent) =>
        onEvent({ type: t, payload: JSON.parse(e.data) }),
      );
    });
    return () => es.close();
  }
}
