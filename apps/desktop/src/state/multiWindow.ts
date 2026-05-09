/**
 * Multi-window helpers (F-D17).
 *
 * Wrap the Tauri ``open_in_new_window`` command with a label
 * generator so reopening the same route lands in a fresh window
 * instead of failing on a duplicate label. Selection state is
 * shared via the localStorage ``storage`` event listener in
 * selection.tsx — this module is just the spawn side.
 *
 * Falls back to ``window.open`` when the Tauri invoke isn't
 * available (e.g. unit tests, dev shells outside Tauri). The
 * fallback opens a real browser window pointed at the same route,
 * which is broken in the dev server but harmless and lets vitest
 * exercise the helper without mocking the whole Tauri stack.
 */
import { invoke } from "@tauri-apps/api/core";

function nextLabel(prefix: string): string {
  // Use crypto.randomUUID so HMR / module reload can't reset a
  // counter into collision territory. Date.now() bucketing is fine
  // for human-readability of the label but isn't sufficient on its
  // own for uniqueness across reloads.
  const uuid = (() => {
    try {
      return globalThis.crypto?.randomUUID?.() ?? "";
    } catch {
      return "";
    }
  })();
  const suffix = uuid ? uuid.slice(0, 8) : Math.random().toString(36).slice(2, 10);
  return `extra-${prefix}-${Date.now()}-${suffix}`;
}

/** Spawn a new window pointed at ``route``. Returns true when the
 * Tauri invoke succeeds, false on fallback or error. The caller
 * shouldn't depend on the return value for UX correctness — the
 * window appears (or doesn't) regardless. */
export async function openInNewWindow(route: string): Promise<boolean> {
  const label = nextLabel(route.replace(/[^a-z0-9]+/gi, "-").slice(0, 16));
  try {
    await invoke("open_in_new_window", { label, route });
    return true;
  } catch {
    try {
      // Browser fallback — the Tauri capability allowlist will refuse
      // the invoke in a dev shell that doesn't include the command.
      window.open(route, "_blank", "noopener,noreferrer");
      return false;
    } catch {
      return false;
    }
  }
}
