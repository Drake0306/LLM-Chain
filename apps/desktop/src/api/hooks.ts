import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";

import { ApiClient } from "./client";

export interface SidecarStatus {
  client: ApiClient | null;
  /** Number of poll attempts so far. Surfaces "still trying after 5 seconds"
   * messaging in the UI when the sidecar is slow to start. */
  attempts: number;
  /** True after we've polled long enough that the user should suspect
   * something is wrong rather than a routine slow boot. */
  slow: boolean;
}

const POLL_INTERVAL_MS = 100;
// First-second threshold for "slow boot" UI affordance. Cold Python imports
// on a freshly-opened laptop can take 2–4 s; anything past 5 s means the
// sidecar has actually failed to start and the user needs to know.
const SLOW_AFTER_MS = 5_000;

export function useApiClient(): ApiClient | null {
  return useSidecarStatus().client;
}

/** Like useApiClient but exposes loading progress so the Dashboard can
 * distinguish "still booting" from "stuck" instead of staring at "Probing
 * hardware…" indefinitely. */
export function useSidecarStatus(): SidecarStatus {
  const [client, setClient] = useState<ApiClient | null>(null);
  const [attempts, setAttempts] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      // Retry forever (with a short interval). The previous 50-iteration
      // cap silently gave up after 5 s — past that point the UI showed
      // "Probing hardware…" with no way to recover short of a relaunch,
      // even though the sidecar might come up moments later.
      let i = 0;
      while (!cancelled) {
        i += 1;
        setAttempts(i);
        try {
          const port = await invoke<number | null>("sidecar_port");
          if (port) {
            setClient(new ApiClient(port));
            return;
          }
        } catch {
          // Tauri command failed (sidecar process killed, dev shell down).
          // Keep polling — recovery is the user's call.
        }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const slow = !client && attempts * POLL_INTERVAL_MS >= SLOW_AFTER_MS;
  return { client, attempts, slow };
}
