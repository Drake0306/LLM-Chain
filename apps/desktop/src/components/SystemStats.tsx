import { useEffect, useState } from "react";

import type { SystemStats as Stats } from "../api/client";
import { useApiClient } from "../api/hooks";

const POLL_MS = 2000;

export function SystemStats() {
  const api = useApiClient();
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    if (!api) return;
    let alive = true;
    const tick = () => {
      api
        .getSystemStats()
        .then((s) => {
          if (alive) setStats(s);
        })
        .catch(() => undefined);
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [api]);

  if (!stats) return null;

  return (
    <div className="flex items-center gap-3 text-xs text-zinc-600 font-mono">
      <Bar label="CPU" percent={stats.cpu_percent} />
      <Bar
        label="RAM"
        percent={stats.ram.percent}
        sub={`${stats.ram.used_gb.toFixed(1)} / ${stats.ram.total_gb.toFixed(0)} GB`}
      />
      {stats.gpu && (
        <Bar
          label="GPU"
          percent={stats.gpu.vram_percent}
          sub={`${stats.gpu.vram_used_gb.toFixed(1)} / ${stats.gpu.vram_total_gb.toFixed(0)} GB`}
          title={stats.gpu.name}
        />
      )}
    </div>
  );
}

function Bar({
  label,
  percent,
  sub,
  title,
}: {
  label: string;
  percent: number;
  sub?: string;
  title?: string;
}) {
  const color =
    percent > 90 ? "bg-red-500" : percent > 70 ? "bg-amber-500" : "bg-green-500";
  return (
    <div className="flex items-center gap-1.5" title={title}>
      <span className="uppercase tracking-wide text-zinc-400 w-7">{label}</span>
      <div className="w-12 h-1.5 bg-zinc-200 rounded-full overflow-hidden">
        <div
          className={`h-full transition-[width] duration-300 ${color}`}
          style={{ width: `${Math.min(100, percent)}%` }}
        />
      </div>
      <span className="tabular-nums w-9 text-right">{Math.round(percent)}%</span>
      {sub && <span className="text-zinc-400 text-[10px] tabular-nums">{sub}</span>}
    </div>
  );
}
