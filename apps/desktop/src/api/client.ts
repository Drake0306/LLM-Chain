/** Parse one ``event: NAME\ndata: JSON`` block out of an SSE stream.
 * Returns null for keep-alive frames or malformed input. The streaming
 * generate endpoint reads multi-line JSON over SSE so we can't lean on
 * the browser's native EventSource (which is GET-only and would lose
 * us the JSON body). */
function parseSseFrame(frame: string): { event: string; data: any } | null {
  let event = "message";
  let dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

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
  // True when the sidecar saw LLM_CHAIN_ROCM_EXPERIMENTAL=1 at startup —
  // the AMD ROCm card becomes selectable and HfRocmTrainer accepts LoRA runs.
  rocm_experimental_armed?: boolean;
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
  chat_capable: boolean;
}

export interface RunConfig {
  model_id: string;
  backend: string;
  technique: "lora" | "qlora";
  dataset_path: string;
  dataset_format?: string;
  text_column?: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  lora_rank?: number;
  lora_alpha?: number;
  /** Optional parent run id; when set, training resumes from that run's
   * adapter instead of starting from random LoRA init. The sidecar
   * validates compatibility (same model, same rank/alpha, same backend
   * family) before accepting the run. */
  resume_from?: string;
  /** Hard cap on iterations. Used by the LR finder to spawn short
   * sniff runs. Null/absent means use the epoch-based default. */
  max_steps?: number | null;
  /** Tag for special-purpose runs ("lr_finder"). UI uses it to group
   * or hide them from the main Runs list. */
  purpose?: string | null;
  /** F-C10: training method. "sft" (default) drives the existing
   * supervised flows; "dpo" switches HF backends to TRL's DPOTrainer
   * and requires a jsonl_dpo dataset. mlx backends reject dpo. */
  training_method?: "sft" | "dpo";
  /** F-C10: KL-penalty weight for DPO (ignored when training_method
   * != "dpo"). Default 0.1 — lower keeps the policy closer to the
   * reference model. */
  dpo_beta?: number;
}

export interface Run {
  id: string;
  created_at: string;
  status: "pending" | "running" | "succeeded" | "failed" | "canceled";
  config: RunConfig;
  error: string | null;
  output_dir: string | null;
  /** Total bytes of saved adapter weights (sum of
   * adapter_model.safetensors / adapters.safetensors / every file
   * under checkpoint-*\/). Only populated for SUCCEEDED runs by
   * GET /api/runs; null otherwise. The Library page renders this. */
  adapter_size_bytes?: number | null;
}

export interface TrainingEventPayload {
  type: "start" | "step" | "epoch_end" | "download" | "log" | "done" | "error" | "canceled";
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

  async getSystemStats(): Promise<SystemStats> {
    const r = await this.fetchImpl(this.base("/api/system/stats"));
    return r.json();
  }

  async getModels(
    maxParams?: number,
    includeRestricted?: boolean,
    modalities?: string[],
    chatCapable?: boolean,
  ): Promise<{ models: ModelEntry[] }> {
    const params = new URLSearchParams();
    if (maxParams) params.set("max_params", String(maxParams));
    if (includeRestricted) params.set("include_restricted", "1");
    if (modalities && modalities.length > 0) {
      params.set("modalities", modalities.join(","));
    }
    if (chatCapable) params.set("chat_capable", "1");
    const q = params.toString();
    const r = await this.fetchImpl(this.base(`/api/models${q ? `?${q}` : ""}`));
    return r.json();
  }

  async scheduleRun(body: {
    config: RunConfig;
    start_at: string;
    fire_if_missed?: boolean;
  }): Promise<ScheduledEntry> {
    const r = await this.fetchImpl(this.base("/api/runs/schedule"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Schedule failed (${r.status})`);
    }
    return r.json();
  }

  async listScheduledRuns(): Promise<{ scheduled: ScheduledEntry[] }> {
    const r = await this.fetchImpl(this.base("/api/runs/schedule"));
    return r.json();
  }

  async cancelScheduledRun(scheduledId: string): Promise<{ canceled: boolean }> {
    const r = await this.fetchImpl(
      this.base(`/api/runs/schedule/${scheduledId}`),
      { method: "DELETE" },
    );
    if (r.status === 404) return { canceled: false };
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Cancel failed (${r.status})`);
    }
    return r.json();
  }

  async createRun(cfg: RunConfig): Promise<{ id: string; status: string }> {
    const r = await this.fetchImpl(this.base("/api/runs"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    if (!r.ok) {
      // Surface the sidecar's detail message (e.g. "Pythia 70M is a base
      // model with no chat template…") instead of a generic HTTP error.
      const body = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(body.detail ?? `Run creation failed (${r.status})`);
    }
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

  async getRunEvents(runId: string): Promise<{ events: TrainingEventPayload[] }> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/events`));
    return r.json();
  }

  async cancelRun(runId: string): Promise<{ canceled: boolean }> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/cancel`), {
      method: "POST",
    });
    if (r.status === 409) {
      // Nothing to cancel — surface as a soft no-op so callers can just
      // refresh the run state and let the UI reconcile. This is the common
      // "user clicked Cancel after run already finished" case.
      return { canceled: false };
    }
    if (!r.ok) {
      // Non-409 errors (sidecar 5xx, network) used to be silently swallowed
      // — the user clicked Cancel and saw nothing happen with no clue why.
      // Bubble up so RunDetail can show a banner.
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Cancel failed (${r.status})`);
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

  /** Fast row count for the Train page. Doesn't parse rows, so it's
   * cheap on multi-GB JSONLs. Returns null row_count for HF Hub
   * (would require a full download to count). */
  async countDataset(body: {
    dataset_path: string;
    dataset_format: string;
    text_column?: string;
  }): Promise<{ row_count: number | null; format: string }> {
    const r = await this.fetchImpl(this.base("/api/datasets/count"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Count failed (${r.status})`);
    }
    return r.json();
  }

  /** Stream synthesised conversation rows from a base model or
   * trained adapter. Each ``onRow`` payload corresponds to one
   * row in the eventual JSONL — ``parsed`` indicates whether the
   * model's output parsed cleanly into the chat schema; the
   * ``raw_text`` is always populated so the UI can show what came
   * back even when parsing failed. Returns a disposer that aborts
   * the SSE consumer (and the underlying generation). */
  synthDataset(
    body: SynthBody,
    handlers: {
      onRow: (row: SynthRow) => void;
      onStatus?: (msg: string) => void;
      onDone: (stats: SynthDoneStats) => void;
      onError: (msg: string) => void;
    },
  ): () => void {
    const controller = new AbortController();
    (async () => {
      try {
        const r = await this.fetchImpl(this.base("/api/datasets/synth"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        if (!r.ok) {
          const detail = await r.json().catch(() => ({} as { detail?: string }));
          handlers.onError(detail.detail ?? `Synth failed (${r.status})`);
          return;
        }
        if (!r.body) {
          handlers.onError("No response body from synth endpoint.");
          return;
        }
        const reader = r.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let sep = buffer.indexOf("\n\n");
          while (sep !== -1) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            const parsed = parseSseFrame(frame);
            if (parsed) {
              if (parsed.event === "row") {
                handlers.onRow(parsed.data as SynthRow);
              } else if (parsed.event === "status") {
                handlers.onStatus?.(parsed.data.status);
              } else if (parsed.event === "done") {
                handlers.onDone(parsed.data.stats ?? { total: 0, parsed_ok: 0, parse_failed: 0 });
                return;
              } else if (parsed.event === "error") {
                handlers.onError(parsed.data.error ?? "synth failed");
                return;
              }
            }
            sep = buffer.indexOf("\n\n");
          }
        }
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        handlers.onError(String((e as Error).message ?? e));
      }
    })();
    return () => controller.abort();
  }

  /** Stream a multi-adapter chat turn. Each generation in the body
   * has its own per-adapter message history; the stream interleaves
   * token frames tagged with the originating adapter_id. The
   * disposer aborts the SSE consumer (and the in-flight generation
   * via the route's cancel hook). */
  multiChat(
    body: MultiChatBody,
    handlers: {
      onToken: (adapterId: string, text: string) => void;
      onStatus?: (adapterId: string, msg: string) => void;
      onDone: () => void;
      onError: (msg: string) => void;
    },
  ): () => void {
    const controller = new AbortController();
    (async () => {
      try {
        const r = await this.fetchImpl(this.base("/api/chat/multi"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        if (!r.ok) {
          const detail = await r.json().catch(() => ({} as { detail?: string }));
          handlers.onError(detail.detail ?? `Multi-chat failed (${r.status})`);
          return;
        }
        if (!r.body) {
          handlers.onError("No response body from multi-chat endpoint.");
          return;
        }
        const reader = r.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let sep = buffer.indexOf("\n\n");
          while (sep !== -1) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            const parsed = parseSseFrame(frame);
            if (parsed) {
              if (parsed.event === "token") {
                const { adapter_id, text } = parsed.data;
                if (typeof adapter_id === "string" && typeof text === "string") {
                  handlers.onToken(adapter_id, text);
                }
              } else if (parsed.event === "status") {
                handlers.onStatus?.(parsed.data.adapter_id ?? "", parsed.data.status);
              } else if (parsed.event === "done") {
                handlers.onDone();
                return;
              } else if (parsed.event === "error") {
                handlers.onError(parsed.data.error ?? "multi-chat failed");
                return;
              }
            }
            sep = buffer.indexOf("\n\n");
          }
        }
        handlers.onDone();
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        handlers.onError(String((e as Error).message ?? e));
      }
    })();
    return () => controller.abort();
  }

  async registerOllama(
    runId: string,
    body: OllamaRegisterBody,
  ): Promise<OllamaRegisterResult> {
    const r = await this.fetchImpl(
      this.base(`/api/runs/${runId}/ollama/register`),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Register failed (${r.status})`);
    }
    return r.json();
  }

  async listOllamaRegistrations(
    runId: string,
  ): Promise<{ names: string[]; ollama_installed: boolean }> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/ollama`));
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `List failed (${r.status})`);
    }
    return r.json();
  }

  async unregisterOllama(
    runId: string,
    name: string,
  ): Promise<{ unregistered: boolean }> {
    const r = await this.fetchImpl(
      this.base(`/api/runs/${runId}/ollama/${encodeURIComponent(name)}`),
      { method: "DELETE" },
    );
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Unregister failed (${r.status})`);
    }
    return r.json();
  }

  async getLeaderboardConfig(): Promise<{
    configured: boolean;
    endpoint: string | null;
  }> {
    const r = await this.fetchImpl(this.base("/api/leaderboard/config"));
    return r.json();
  }

  async submitToLeaderboard(
    runId: string,
    body: LeaderboardSubmitBody,
  ): Promise<LeaderboardSubmitResult> {
    const r = await this.fetchImpl(
      this.base(`/api/runs/${runId}/leaderboard/submit`),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Submit failed (${r.status})`);
    }
    return r.json();
  }

  async listLeaderboardSubmissions(
    runId: string,
  ): Promise<{
    configured: boolean;
    submissions: LeaderboardSubmission[];
  }> {
    const r = await this.fetchImpl(
      this.base(`/api/runs/${runId}/leaderboard`),
    );
    return r.json();
  }

  async mergeRuns(body: MergeBody): Promise<MergeResult> {
    const r = await this.fetchImpl(this.base("/api/runs/merge"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Merge failed (${r.status})`);
    }
    return r.json();
  }

  async listRecipes(): Promise<{ recipes: Recipe[]; app_version: string }> {
    const r = await this.fetchImpl(this.base("/api/recipes"));
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Recipe list failed (${r.status})`);
    }
    return r.json();
  }

  async listCuratedDatasets(): Promise<{ datasets: CuratedEntry[] }> {
    const r = await this.fetchImpl(this.base("/api/datasets/curated"));
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Curated list failed (${r.status})`);
    }
    return r.json();
  }

  async startCuratedDownload(
    curatedId: string,
  ): Promise<CuratedDownloadState> {
    const r = await this.fetchImpl(
      this.base(
        `/api/datasets/curated/${encodeURIComponent(curatedId)}/download`,
      ),
      { method: "POST" },
    );
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Download start failed (${r.status})`);
    }
    return r.json();
  }

  async getCuratedDownloadStatus(
    curatedId: string,
  ): Promise<CuratedDownloadState | null> {
    const r = await this.fetchImpl(
      this.base(
        `/api/datasets/curated/${encodeURIComponent(curatedId)}/status`,
      ),
    );
    if (r.status === 404) return null;
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Status failed (${r.status})`);
    }
    return r.json();
  }

  async buildDataset(body: DatasetBuildBody): Promise<DatasetBuildResult> {
    const r = await this.fetchImpl(this.base("/api/datasets/build"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Build failed (${r.status})`);
    }
    return r.json();
  }

  async previewDataset(body: {
    dataset_path: string;
    dataset_format: string;
    text_column?: string;
    limit?: number;
  }): Promise<DatasetPreview> {
    const r = await this.fetchImpl(this.base("/api/datasets/preview"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Preview failed (${r.status})`);
    }
    return r.json();
  }

  /** Stream generated tokens for a run's trained adapter. The returned
   * disposer aborts the underlying fetch, which closes the SSE
   * connection on the sidecar side and triggers in-flight cancel.
   *
   * - ``onToken``: decoded delta to append to the visible response.
   * - ``onStatus``: transient UI hint ("Loading model…", "Switching
   *   cached model…"). Not appended to model output.
   * - ``onDone``: clean end of stream.
   * - ``onError``: transport error or sidecar-surfaced error event.
   */
  generateRun(
    runId: string,
    body: { prompt: string; max_tokens?: number; temperature?: number; top_p?: number },
    handlers: {
      onToken: (text: string) => void;
      onStatus?: (msg: string) => void;
      onDone: () => void;
      onError: (msg: string) => void;
    },
  ): () => void {
    // EventSource is GET-only and our endpoint is POST. Use fetch +
    // a manual SSE parser instead — the response body is text streamed
    // line-by-line in `event: <name>\ndata: <json>\n\n` blocks.
    const controller = new AbortController();
    (async () => {
      try {
        const r = await this.fetchImpl(this.base(`/api/runs/${runId}/generate`), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        if (!r.ok) {
          const detail = await r.json().catch(() => ({} as { detail?: string }));
          handlers.onError(detail.detail ?? `Generate failed (${r.status})`);
          return;
        }
        if (!r.body) {
          handlers.onError("No response body from generate endpoint.");
          return;
        }
        const reader = r.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        // Each SSE frame is two newlines apart. Decode incrementally
        // so a multi-byte UTF-8 token split across reads doesn't show
        // up as a replacement char on the wire.
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let sep = buffer.indexOf("\n\n");
          while (sep !== -1) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            const parsed = parseSseFrame(frame);
            if (parsed) {
              if (parsed.event === "token" && typeof parsed.data.text === "string") {
                handlers.onToken(parsed.data.text);
              } else if (
                parsed.event === "status" &&
                typeof parsed.data.status === "string"
              ) {
                handlers.onStatus?.(parsed.data.status);
              } else if (parsed.event === "done") {
                handlers.onDone();
                return;
              } else if (parsed.event === "error") {
                handlers.onError(parsed.data.error ?? "generate failed");
                return;
              }
            }
            sep = buffer.indexOf("\n\n");
          }
        }
        handlers.onDone();
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        handlers.onError(String((e as Error).message ?? e));
      }
    })();
    return () => controller.abort();
  }

  /** Skip the current prompt in an in-flight eval. The orchestrator
   * advances to the next prompt at the next token boundary. Returns
   * canceled=false when there's no eval running (409) so callers
   * can treat the click as a soft no-op. */
  async skipEvalPrompt(runId: string): Promise<{ signaled: boolean }> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/eval/skip`), {
      method: "POST",
    });
    if (r.status === 409) return { signaled: false };
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Skip failed (${r.status})`);
    }
    return r.json();
  }

  /** Family-aware default eval prompts. Eval screen calls this on
   * mount to prefill the textarea with prompts relevant to the run's
   * model family rather than the generic placeholders. */
  async getEvalDefaults(
    runId: string,
  ): Promise<{ family: string | null; prompts: string[] }> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/eval/defaults`));
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Eval defaults failed (${r.status})`);
    }
    return r.json();
  }

  /** Stream side-by-side base/adapter outputs for an eval suite.
   * The handlers parallel ``generateRun`` plus an ``onEval`` for
   * the per-token frames that carry a (role, prompt_index, text)
   * triple. Returns a disposer that aborts the underlying fetch. */
  evalRun(
    runId: string,
    body: { prompts: string[]; max_tokens?: number; temperature?: number },
    handlers: {
      onEval: (role: "base" | "adapter", promptIndex: number, text: string) => void;
      onStatus?: (msg: string) => void;
      onDone: () => void;
      onError: (msg: string) => void;
    },
  ): () => void {
    const controller = new AbortController();
    (async () => {
      try {
        const r = await this.fetchImpl(this.base(`/api/runs/${runId}/eval`), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        if (!r.ok) {
          const detail = await r.json().catch(() => ({} as { detail?: string }));
          handlers.onError(detail.detail ?? `Eval failed (${r.status})`);
          return;
        }
        if (!r.body) {
          handlers.onError("No response body from eval endpoint.");
          return;
        }
        const reader = r.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let sep = buffer.indexOf("\n\n");
          while (sep !== -1) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            const parsed = parseSseFrame(frame);
            if (parsed) {
              if (parsed.event === "token") {
                const { role, prompt_index, text } = parsed.data;
                if (
                  (role === "base" || role === "adapter") &&
                  typeof prompt_index === "number" &&
                  typeof text === "string"
                ) {
                  handlers.onEval(role, prompt_index, text);
                }
              } else if (
                parsed.event === "status" &&
                typeof parsed.data.status === "string"
              ) {
                handlers.onStatus?.(parsed.data.status);
              } else if (parsed.event === "done") {
                handlers.onDone();
                return;
              } else if (parsed.event === "error") {
                handlers.onError(parsed.data.error ?? "eval failed");
                return;
              }
            }
            sep = buffer.indexOf("\n\n");
          }
        }
        handlers.onDone();
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        handlers.onError(String((e as Error).message ?? e));
      }
    })();
    return () => controller.abort();
  }

  /** Stream side-by-side outputs for two adapters on a shared
   * prompt set. Wire format matches ``evalRun`` but with ``role``
   * being "left" / "right" instead of "base" / "adapter". The
   * disposer aborts the underlying SSE stream — caller's responsibility
   * to call it on unmount. */
  comparePrompts(
    body: {
      left_run_id: string;
      right_run_id: string;
      prompts: string[];
      max_tokens?: number;
      temperature?: number;
    },
    handlers: {
      onToken: (role: "left" | "right", promptIndex: number, text: string) => void;
      onStatus?: (msg: string) => void;
      onDone: () => void;
      onError: (msg: string) => void;
    },
  ): () => void {
    const controller = new AbortController();
    (async () => {
      try {
        const r = await this.fetchImpl(this.base("/api/compare/prompts"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        if (!r.ok) {
          const detail = await r.json().catch(() => ({} as { detail?: string }));
          handlers.onError(detail.detail ?? `Compare failed (${r.status})`);
          return;
        }
        if (!r.body) {
          handlers.onError("No response body from compare endpoint.");
          return;
        }
        const reader = r.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let sep = buffer.indexOf("\n\n");
          while (sep !== -1) {
            const frame = buffer.slice(0, sep);
            buffer = buffer.slice(sep + 2);
            const parsed = parseSseFrame(frame);
            if (parsed) {
              if (parsed.event === "token") {
                const { role, prompt_index, text } = parsed.data;
                if (
                  (role === "left" || role === "right") &&
                  typeof prompt_index === "number" &&
                  typeof text === "string"
                ) {
                  handlers.onToken(role, prompt_index, text);
                }
              } else if (parsed.event === "status") {
                handlers.onStatus?.(parsed.data.status);
              } else if (parsed.event === "done") {
                handlers.onDone();
                return;
              } else if (parsed.event === "error") {
                handlers.onError(parsed.data.error ?? "compare failed");
                return;
              }
            }
            sep = buffer.indexOf("\n\n");
          }
        }
        handlers.onDone();
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        handlers.onError(String((e as Error).message ?? e));
      }
    })();
    return () => controller.abort();
  }

  async skipComparePrompt(
    leftRunId: string,
    rightRunId: string,
  ): Promise<{ signaled: boolean }> {
    const params = new URLSearchParams({
      left_run_id: leftRunId,
      right_run_id: rightRunId,
    });
    const r = await this.fetchImpl(
      this.base(`/api/compare/skip?${params.toString()}`),
      { method: "POST" },
    );
    if (r.status === 409) return { signaled: false };
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Skip failed (${r.status})`);
    }
    return r.json();
  }

  async resumeRun(
    runId: string,
    body: { epochs: number; learning_rate?: number },
  ): Promise<{ id: string; status: string }> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/resume`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Resume failed (${r.status})`);
    }
    return r.json();
  }

  async lrFinder(body: {
    config: RunConfig;
    learning_rates: number[];
    steps_per_run?: number;
  }): Promise<{ run_ids: string[]; steps_per_run: number }> {
    const r = await this.fetchImpl(this.base("/api/runs/lr-finder"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `LR finder failed (${r.status})`);
    }
    return r.json();
  }

  async cleanupRuns(body: {
    older_than_days?: number;
    statuses?: ("failed" | "canceled" | "succeeded")[];
  }): Promise<{ deleted_count: number; freed_bytes: number; deleted_ids: string[] }> {
    const r = await this.fetchImpl(this.base("/api/maintenance/cleanup"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Cleanup failed (${r.status})`);
    }
    return r.json();
  }

  async getRunNotes(runId: string): Promise<{ markdown: string }> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/notes`));
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Notes load failed (${r.status})`);
    }
    return r.json();
  }

  async putRunNotes(
    runId: string,
    markdown: string,
  ): Promise<{ saved: boolean; bytes: number }> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/notes`), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ markdown }),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Notes save failed (${r.status})`);
    }
    return r.json();
  }

  async deleteRun(runId: string): Promise<{ deleted: boolean }> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}`), {
      method: "DELETE",
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { detail?: string }));
      throw new Error(detail.detail ?? `Delete failed (${r.status})`);
    }
    return r.json();
  }

  async getHubExport(runId: string): Promise<HubPushState | null> {
    const r = await this.fetchImpl(this.base(`/api/runs/${runId}/export/hub`));
    if (r.status === 404) return null;
    return r.json();
  }

  async pushRunToHub(
    runId: string,
    body: { repo_id: string; private: boolean; folder?: "adapter" | "merged" },
  ): Promise<HubPushState> {
    // Returns the initial 202 state. The caller polls getHubExport() for
    // progress + final URL. Pre-flight failures (auth, bad folder)
    // come back as 4xx with a detail message; anything else means the
    // background worker took over.
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
      "log",
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

export interface SystemStats {
  cpu_percent: number;
  ram: { used_gb: number; total_gb: number; percent: number };
  gpu: {
    name: string;
    vram_used_gb: number;
    vram_total_gb: number;
    vram_percent: number;
  } | null;
}

export type GgufQuant = "q4_k_m" | "q8_0" | "f16";

export interface GgufExportState {
  status: "running" | "done" | "failed";
  step?: "merge" | "convert";
  quant: GgufQuant | string;
  path?: string;
  merged_path?: string | null;
  error?: string;
  latest_log?: string | null;
}

export interface HubPushResult {
  url: string;
  repo_id: string;
  private: boolean;
}

export interface HubPushState {
  status: "running" | "done" | "failed";
  repo_id?: string;
  private?: boolean;
  folder?: string;
  url?: string;
  error?: string;
  error_kind?: "auth" | "missing" | "invalid" | "unknown";
  latest_log?: string | null;
}

export interface DatasetPreview {
  /** Parsed rows up to the requested limit. Shape varies by dataset
   * format — chat rows have ``messages``, plain-text rows have
   * ``text``, etc. The UI just renders them as JSON. */
  rows: unknown[];
  row_count: number;
  shown: number;
}

export interface DatasetBuildBody {
  /** Exactly one of raw_text or source_path must be set. */
  raw_text?: string;
  source_path?: string;
  input_format: "csv" | "tsv" | "jsonl";
  target?: "chat" | "completion";
  user_field?: string;
  assistant_field?: string;
  prompt_field?: string;
  completion_field?: string;
  passthrough_chat?: boolean;
  drop_empty?: boolean;
  dedupe?: boolean;
  role_balance?: boolean;
  max_chars?: number | null;
  /** Filename stem; the workshop appends .jsonl under ~/.llm-chain/datasets/. */
  name?: string;
  /** Full output path override; takes precedence over ``name``. */
  output_path?: string;
}

export interface DatasetBuildStats {
  input_rows: number;
  dropped_empty: number;
  dropped_duplicate: number;
  dropped_role_violation: number;
  dropped_length: number;
  output_rows: number;
}

export interface DatasetBuildResult {
  path: string;
  bytes_written: number;
  stats: DatasetBuildStats;
}

export interface SynthBody {
  source_run_id?: string;
  source_model_id?: string;
  source_backend?: string;
  topic: string;
  style?: string;
  count?: number;
  max_tokens?: number;
  temperature?: number;
  seed_prompts?: string[];
}

export interface SynthRow {
  index: number;
  /** Parsed messages list when the model's output round-tripped
   * through the chat-shape validator. Null on parse failure — UI
   * shows raw_text so the user can still see what came back. */
  messages: { role: string; content: string }[] | null;
  raw_text: string | null;
  parsed: boolean;
}

export interface LeaderboardSubmitBody {
  repo_id: string;
  eval_results?: Record<string, unknown>;
  notes?: string;
  license_name?: string;
}

export interface LeaderboardSubmitResult {
  endpoint: string;
  payload: Record<string, unknown>;
  response: Record<string, unknown>;
}

export interface LeaderboardSubmission {
  endpoint: string;
  payload: Record<string, unknown>;
  response: Record<string, unknown>;
}

export interface MergeAdapterEntry {
  run_id: string;
  weight: number;
}

export type MergeMethod = "linear" | "ties" | "dare";

export interface MergeBody {
  runs: MergeAdapterEntry[];
  method: MergeMethod;
  method_options?: Record<string, number | string | boolean>;
}

export interface MergeResult {
  id: string;
  method: MergeMethod;
  sources: string[];
  weights: number[];
  output_dir: string;
}

export interface MultiChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface MultiChatGeneration {
  run_id: string;
  messages: MultiChatMessage[];
}

export interface MultiChatBody {
  generations: MultiChatGeneration[];
  max_tokens?: number;
  temperature?: number;
}

export interface OllamaRegisterBody {
  name: string;
  temperature?: number;
  top_p?: number;
  num_ctx?: number;
  stop_tokens?: string[];
  system?: string | null;
}

export interface OllamaRegisterResult {
  name: string;
  run_command: string;
  modelfile_path: string;
}

export interface RecipeDataset {
  kind: "curated" | "synth" | "bring_your_own" | "unknown";
  curated_id: string | null;
  synth_topic: string | null;
  synth_style: string | null;
  bring_your_own: boolean;
}

export interface RecipeHyperparameters {
  epochs: number;
  learning_rate: number;
  lora_rank: number;
  lora_alpha: number;
  batch_size: number;
}

export interface Recipe {
  id: string;
  name: string;
  description: string;
  model: string;
  technique: "lora" | "qlora";
  dataset: RecipeDataset;
  hyperparameters: RecipeHyperparameters;
  min_app_version: string;
  suggested_backend: string | null;
  notes: string;
  needs_upgrade: boolean;
}

export interface CuratedEntry {
  id: string;
  name: string;
  hf_id: string;
  description: string;
  license: string;
  license_url: string;
  size_rows: number;
  size_mb: number;
  schema: string;
  suitable_for: string[];
}

export interface CuratedDownloadState {
  id?: string;
  status: "running" | "done" | "failed";
  path?: string;
  rows_loaded?: number;
  rows_kept?: number;
  error?: string;
  error_kind?: string;
}

export interface ScheduledEntry {
  id: string;
  start_at: string; // ISO
  fire_if_missed: boolean;
  config: RunConfig;
  created_at: string;
}

export interface SynthDoneStats {
  total: number;
  parsed_ok: number;
  parse_failed: number;
}
