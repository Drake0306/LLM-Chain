import { describe, expect, it, vi } from "vitest";

import { ApiClient } from "../client";

describe("ApiClient", () => {
  it("constructs URLs against the resolved sidecar port", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ os: "Darwin", devices: [] }),
    });
    const c = new ApiClient(8123, fetchMock as unknown as typeof fetch);
    const r = await c.getHardware();
    expect(fetchMock).toHaveBeenCalledWith("http://127.0.0.1:8123/api/hardware");
    expect(r.os).toBe("Darwin");
  });

  it("getModels appends max_params when provided", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [] }),
    });
    const c = new ApiClient(8123, fetchMock as unknown as typeof fetch);
    await c.getModels(500_000_000);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8123/api/models?max_params=500000000",
    );
  });

  it("getModels forwards include_restricted when true", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [] }),
    });
    const c = new ApiClient(8123, fetchMock as unknown as typeof fetch);
    await c.getModels(500_000_000, true);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8123/api/models?max_params=500000000&include_restricted=1",
    );
  });

  it("getModels omits include_restricted when false", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [] }),
    });
    const c = new ApiClient(8123, fetchMock as unknown as typeof fetch);
    await c.getModels(undefined, false);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8123/api/models",
    );
  });

  it("getModels forwards modalities as a CSV", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [] }),
    });
    const c = new ApiClient(8123, fetchMock as unknown as typeof fetch);
    await c.getModels(undefined, false, ["text", "image"]);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8123/api/models?modalities=text%2Cimage",
    );
  });

  it("getModels omits modalities when the array is empty", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [] }),
    });
    const c = new ApiClient(8123, fetchMock as unknown as typeof fetch);
    await c.getModels(undefined, false, []);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8123/api/models",
    );
  });

  it("createRun POSTs the config", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ id: "abc", status: "pending" }),
    });
    const c = new ApiClient(8123, fetchMock as unknown as typeof fetch);
    const r = await c.createRun({
      model_id: "Qwen/Qwen3-0.6B",
      backend: "cuda",
      technique: "lora",
      dataset_path: "/tmp/x",
      epochs: 1,
    });
    expect(r.id).toBe("abc");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8123/api/runs",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
