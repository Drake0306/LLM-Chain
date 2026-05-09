import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ensurePermission, notify, permission } from "../notifications";

class FakeNotification {
  static permission: NotificationPermission = "default";
  static instances: FakeNotification[] = [];
  static requestPermission = vi.fn(async () =>
    FakeNotification.permission,
  );

  constructor(public title: string, public options: NotificationOptions = {}) {
    FakeNotification.instances.push(this);
  }
}

beforeEach(() => {
  FakeNotification.permission = "default";
  FakeNotification.instances = [];
  FakeNotification.requestPermission = vi.fn(async () =>
    FakeNotification.permission,
  );
  // @ts-expect-error — wire the fake into the global slot vitest's
  // jsdom environment leaves bare; tests reset it in afterEach.
  globalThis.Notification = FakeNotification;
});

afterEach(() => {
  // @ts-expect-error
  delete globalThis.Notification;
});

describe("notifications.permission", () => {
  it("returns 'unsupported' when Notification is missing", () => {
    // @ts-expect-error
    delete globalThis.Notification;
    expect(permission()).toBe("unsupported");
  });

  it("mirrors the Notification.permission value", () => {
    FakeNotification.permission = "granted";
    expect(permission()).toBe("granted");
    FakeNotification.permission = "denied";
    expect(permission()).toBe("denied");
  });
});

describe("notifications.ensurePermission", () => {
  it("short-circuits to 'unsupported' when API is missing", async () => {
    // @ts-expect-error
    delete globalThis.Notification;
    expect(await ensurePermission()).toBe("unsupported");
  });

  it("returns granted without re-asking when already granted", async () => {
    FakeNotification.permission = "granted";
    FakeNotification.requestPermission = vi.fn();
    expect(await ensurePermission()).toBe("granted");
    expect(FakeNotification.requestPermission).not.toHaveBeenCalled();
  });

  it("returns denied without re-asking when already denied", async () => {
    FakeNotification.permission = "denied";
    FakeNotification.requestPermission = vi.fn();
    expect(await ensurePermission()).toBe("denied");
    expect(FakeNotification.requestPermission).not.toHaveBeenCalled();
  });

  it("requests permission only on the default state", async () => {
    FakeNotification.permission = "default";
    FakeNotification.requestPermission = vi.fn(async () => "granted");
    expect(await ensurePermission()).toBe("granted");
    expect(FakeNotification.requestPermission).toHaveBeenCalledOnce();
  });

  it("falls back to denied if requestPermission throws", async () => {
    FakeNotification.permission = "default";
    FakeNotification.requestPermission = vi.fn(async () => {
      throw new Error("blocked by the OS");
    });
    expect(await ensurePermission()).toBe("denied");
  });
});

describe("notifications.notify", () => {
  it("returns false when API is missing", () => {
    // @ts-expect-error
    delete globalThis.Notification;
    const sent = notify({
      status: "succeeded",
      runId: "0123456789ab",
      modelId: "qwen3-1.7b",
    });
    expect(sent).toBe(false);
  });

  it("returns false when permission isn't granted", () => {
    FakeNotification.permission = "default";
    const sent = notify({ status: "succeeded", runId: "x" });
    expect(sent).toBe(false);
    expect(FakeNotification.instances).toHaveLength(0);
  });

  it("fires a notification with title and body when granted", () => {
    FakeNotification.permission = "granted";
    const sent = notify({
      status: "succeeded",
      runId: "0123456789ab",
      modelId: "qwen3-1.7b",
    });
    expect(sent).toBe(true);
    expect(FakeNotification.instances).toHaveLength(1);
    const n = FakeNotification.instances[0];
    expect(n.title).toBe("Training finished");
    // Body includes the model id + truncated run id (first 8 chars).
    expect(n.options.body).toContain("qwen3-1.7b");
    expect(n.options.body).toContain("01234567");
  });

  it("uses status-specific titles", () => {
    FakeNotification.permission = "granted";
    notify({ status: "failed", runId: "abc" });
    notify({ status: "canceled", runId: "abc" });
    expect(FakeNotification.instances[0].title).toBe("Training failed");
    expect(FakeNotification.instances[1].title).toBe("Training canceled");
  });

  it("uses run-id+status tag so cross-status events don't silently overwrite", () => {
    FakeNotification.permission = "granted";
    notify({ status: "succeeded", runId: "abc123def456" });
    notify({ status: "failed", runId: "abc123def456" });
    expect(FakeNotification.instances[0].options.tag).toBe(
      "llm-chain-run-abc123def456-succeeded",
    );
    expect(FakeNotification.instances[1].options.tag).toBe(
      "llm-chain-run-abc123def456-failed",
    );
  });

  it("appends detail line when supplied", () => {
    FakeNotification.permission = "granted";
    notify({
      status: "failed",
      runId: "abc",
      detail: "out of memory at step 12",
    });
    expect(FakeNotification.instances[0].options.body).toContain(
      "out of memory",
    );
  });
});
