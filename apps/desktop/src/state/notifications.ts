/**
 * System notifications via the Web Notifications API (F-D16).
 *
 * The WebView in Tauri 2 supports the standard Notifications API on
 * macOS / Windows / Linux, so we don't need a separate Tauri plugin.
 * Permission is requested lazily on first use; if the user denied
 * the prompt we silently no-op rather than annoying them with
 * another ask.
 *
 * Visibility gate: the caller (RunDetail) only invokes ``notify``
 * when ``document.hidden === true``, so the user with the window in
 * focus doesn't see a redundant popup duplicating what the UI
 * already shows. The functions here are agnostic to that — keeping
 * the policy in the caller makes it testable without a fake
 * Notification global.
 */

export type TerminalRunStatus = "succeeded" | "failed" | "canceled";

const TITLE_BY_STATUS: Record<TerminalRunStatus, string> = {
  succeeded: "Training finished",
  failed: "Training failed",
  canceled: "Training canceled",
};

/** Browser permission values plus a synthetic "unsupported" for envs
 * where the Notification global isn't present (jsdom, very old
 * WebViews). Lets the caller branch on a single value rather than
 * checking globals separately. */
export type NotificationPermissionState =
  | "granted"
  | "denied"
  | "default"
  | "unsupported";

export function permission(): NotificationPermissionState {
  if (typeof Notification === "undefined") return "unsupported";
  return Notification.permission as NotificationPermissionState;
}

/** Idempotent request — once denied, we stop asking. The browser
 * itself enforces this for most platforms but doing it here too
 * means the caller doesn't have to track state. */
export async function ensurePermission(): Promise<NotificationPermissionState> {
  if (typeof Notification === "undefined") return "unsupported";
  if (Notification.permission === "granted") return "granted";
  if (Notification.permission === "denied") return "denied";
  try {
    const result = await Notification.requestPermission();
    return result as NotificationPermissionState;
  } catch {
    return "denied";
  }
}

export interface NotifyOptions {
  status: TerminalRunStatus;
  runId: string;
  modelId?: string;
  detail?: string;
}

/** Fire one notification. Returns true when delivered, false when
 * permission isn't granted or the API isn't available. The caller
 * uses the boolean to decide whether to fall back to an in-app
 * banner; without a return value, a denied permission would pass
 * silently. */
export function notify(opts: NotifyOptions): boolean {
  if (typeof Notification === "undefined") return false;
  if (Notification.permission !== "granted") return false;
  const title = TITLE_BY_STATUS[opts.status];
  const body = opts.modelId
    ? `${opts.modelId} · run ${opts.runId.slice(0, 8)}`
    : `Run ${opts.runId.slice(0, 8)}`;
  try {
    // The constructor's side-effect (queue + show the notification)
    // is what we want; we don't need the instance handle.
    new Notification(title, {
      body: opts.detail ? `${body}\n${opts.detail}` : body,
      tag: `llm-chain-run-${opts.runId}`,
      // requireInteraction false = auto-dismiss after the OS's
      // default timeout. Training results aren't worth holding the
      // notification open until clicked.
    });
    return true;
  } catch {
    return false;
  }
}
