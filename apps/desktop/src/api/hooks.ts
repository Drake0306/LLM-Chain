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
  /** Distinguishes the three lifecycle states the UI cares about:
   *   - "booting": no port yet, keep showing the loading state
   *   - "ready": port is up, client returns a valid ApiClient
   *   - "dead": we had a port at one point but the Tauri side cleared
   *     it (the sidecar process exited). UI should show a recovery
   *     banner instead of letting fetches connect-refuse silently.
   */
  phase: "booting" | "ready" | "dead";
}

const BOOT_POLL_INTERVAL_MS = 100;
// First-second threshold for "slow boot" UI affordance. Cold Python imports
// on a freshly-opened laptop can take 2–4 s; anything past 5 s means the
// sidecar has actually failed to start and the user needs to know.
const SLOW_AFTER_MS = 5_000;
// Lifecheck cadence once the client is up. Cheap (one Tauri invoke) and
// small enough that a sidecar crash surfaces in the UI within a few
// seconds without saturating the IPC channel.
const LIFECHECK_INTERVAL_MS = 2_000;

export function useApiClient(): ApiClient | null {
  return useSidecarStatus().client;
}

/** Tracks the sidecar lifecycle from boot through ready through death.
 *
 * On mount, polls every {@link BOOT_POLL_INTERVAL_MS} until
 * ``invoke("sidecar_port")`` returns a port. After that, drops to a
 * {@link LIFECHECK_INTERVAL_MS} cadence so we notice when the Rust side
 * clears the port (the side-effect of CommandEvent::Terminated). The UI
 * uses ``phase`` to render: spinner during boot, full screen during
 * ready, error banner once dead.
 */
export function useSidecarStatus(): SidecarStatus {
  const [client, setClient] = useState<ApiClient | null>(null);
  const [attempts, setAttempts] = useState(0);
  const [dead, setDead] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    async function pollUntilReady(): Promise<number | null> {
      let i = 0;
      while (!cancelled) {
        i += 1;
        setAttempts(i);
        try {
          const port = await invoke<number | null>("sidecar_port");
          if (port) return port;
        } catch {
          // Tauri command failed (sidecar process killed, dev shell down).
          // Keep polling — recovery is the user's call.
        }
        await new Promise((r) => setTimeout(r, BOOT_POLL_INTERVAL_MS));
      }
      return null;
    }

    function startLifecheck() {
      // Switch from busy-poll to a relaxed interval. If the port ever
      // goes back to null (Rust cleared it on Terminated), flip into
      // dead state. Don't auto-recover — the sidecar binary doesn't
      // currently respawn itself, and silently rebinding to a new port
      // mid-session would mask real crashes from the user.
      intervalId = setInterval(async () => {
        if (cancelled) return;
        try {
          const port = await invoke<number | null>("sidecar_port");
          if (!port) {
            setDead(true);
            setClient(null);
            if (intervalId) {
              clearInterval(intervalId);
              intervalId = null;
            }
          }
        } catch {
          // Same rationale as during boot — keep checking.
        }
      }, LIFECHECK_INTERVAL_MS);
    }

    (async () => {
      const port = await pollUntilReady();
      if (cancelled || !port) return;
      setClient(new ApiClient(port));
      setDead(false);
      startLifecheck();
    })();

    return () => {
      cancelled = true;
      if (intervalId) clearInterval(intervalId);
    };
  }, []);

  const slow = !client && !dead && attempts * BOOT_POLL_INTERVAL_MS >= SLOW_AFTER_MS;
  const phase: SidecarStatus["phase"] = dead
    ? "dead"
    : client
      ? "ready"
      : "booting";
  return { client, attempts, slow, phase };
}
