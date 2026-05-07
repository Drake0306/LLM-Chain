import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";

import { ApiClient } from "./client";

export function useApiClient(): ApiClient | null {
  const [client, setClient] = useState<ApiClient | null>(null);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (let i = 0; i < 50 && !cancelled; i++) {
        const port = await invoke<number | null>("sidecar_port");
        if (port) {
          setClient(new ApiClient(port));
          return;
        }
        await new Promise((r) => setTimeout(r, 100));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);
  return client;
}
