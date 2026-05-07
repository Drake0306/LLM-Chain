export type MemoryKind = "dedicated" | "unified" | "shared";

export interface DeviceCapabilities {
  qlora_max_params: number;
  lora_max_params: number;
  full_ft_max_params: number;
  cpu_max_params: number;
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
  restricted: boolean;
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
  type: "start" | "step" | "epoch_end" | "download" | "done" | "error" | "canceled";
  step: number;
  total_steps: number;
  epoch: number;
  loss: number | null;
  lr: number | null;
  message: string | null;
  bytes_done: number | null;
  bytes_total: number | null;
}

export class ApiClient {
  private fetchImpl: typeof fetch;

  constructor(private port: number, fetchImpl?: typeof fetch) {
    // Native browser fetch enforces `this === Window`. Calling
    // `this.fetchImpl(...)` rebinds `this` to the ApiClient instance and
    // throws "Can only call Window.fetch on instances of Window". Bind once
    // here so call sites can use the natural method-style syntax.
    this.fetchImpl = fetchImpl ?? globalThis.fetch.bind(globalThis);
  }

  private base(path: string) {
    return `http://127.0.0.1:${this.port}${path}`;
  }

  async getHardware(): Promise<HardwareReport> {
    const r = await this.fetchImpl(this.base("/api/hardware"));
    return r.json();
  }

  async getModels(
    maxParams?: number,
    includeRestricted?: boolean,
  ): Promise<{ models: ModelEntry[] }> {
    const params = new URLSearchParams();
    if (maxParams) params.set("max_params", String(maxParams));
    if (includeRestricted) params.set("include_restricted", "1");
    const q = params.toString();
    const r = await this.fetchImpl(this.base(`/api/models${q ? `?${q}` : ""}`));
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

  async cancelRun(runId: string): Promise<{ canceled: boolean }> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/cancel`), {
      method: "POST",
    });
    if (!r.ok) {
      // 409 when nothing to cancel — surface as a soft no-op so callers can
      // just refresh the run state and let the UI reconcile.
      return { canceled: false };
    }
    return r.json();
  }

  async startGgufExport(runId: string, quant: string): Promise<GgufExportState> {
    const r = await this.fetchImpl(
      this.base(`/api/runs/${runId}/export/gguf?quant=${encodeURIComponent(quant)}`),
      { method: "POST" },
    );
    if (!r.ok) {
      const detail = await r.json().catch(() => ({}));
      throw new Error(detail.detail ?? `Export request failed (${r.status})`);
    }
    return r.json();
  }

  async getGgufExport(runId: string): Promise<GgufExportState | null> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/export/gguf`));
    if (r.status === 404) return null;
    return r.json();
  }

  async getHfAuth(): Promise<{ signed_in: boolean }> {
    const r = await this.fetchImpl(this.base("/api/auth/hf"));
    return r.json();
  }

  async pushRunToHub(
    runId: string,
    body: { repo_id: string; private: boolean; folder?: "adapter" | "merged" },
  ): Promise<HubPushResult> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/export/hub`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder: "adapter", ...body }),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({ detail: r.statusText }));
      const err = new Error(detail.detail ?? `Push failed (${r.status})`) as Error & {
        status: number;
      };
      err.status = r.status;
      throw err;
    }
    return r.json();
  }

  streamRun(
    runId: string,
    onEvent: (ev: { type: string; payload: TrainingEventPayload }) => void,
    onState?: (state: StreamState) => void,
  ): () => void {
    const es = new EventSource(this.base(`/api/runs/${runId}/stream`));
    let last: StreamState = "connecting";
    const setState = (s: StreamState) => {
      if (s === last) return;
      last = s;
      onState?.(s);
    };
    es.addEventListener("open", () => setState("open"));
    // EventSource fires "error" both for transient drops (browser will
    // auto-reconnect, readyState === CONNECTING) and for permanent closes
    // (readyState === CLOSED). We surface the difference so the UI can show
    // a "reconnecting…" indicator without panicking the user on normal
    // end-of-stream.
    es.addEventListener("error", () => {
      if (es.readyState === EventSource.CLOSED) {
        setState("closed");
      } else {
        setState("reconnecting");
      }
    });
    const types = [
      "start",
      "step",
      "epoch_end",
      "download",
      "done",
      "error",
      "canceled",
    ] as const;
    types.forEach((t) => {
      es.addEventListener(t, (e: MessageEvent) =>
        onEvent({ type: t, payload: JSON.parse(e.data) }),
      );
    });
    return () => {
      setState("closed");
      es.close();
    };
  }
}

export type StreamState = "connecting" | "open" | "reconnecting" | "closed";

export type GgufQuant = "q4_k_m" | "q8_0" | "f16";

export interface GgufExportState {
  status: "running" | "done" | "failed";
  step?: "merge" | "convert";
  quant: GgufQuant | string;
  path?: string;
  error?: string;
}

export interface HubPushResult {
  url: string;
  repo_id: string;
  private: boolean;
}
