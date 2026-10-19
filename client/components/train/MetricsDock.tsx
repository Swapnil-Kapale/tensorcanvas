"use client";

import { useEffect, useRef } from "react";
import { useTrain, type EpochMetric } from "@/lib/trainStore";

function Tile({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex flex-col rounded-lg bg-neutral-800/50 px-3 py-1.5 ring-1 ring-inset ring-neutral-800">
      <span className="text-[10px] uppercase tracking-wide text-neutral-500">{label}</span>
      <span className={`font-mono text-sm font-semibold ${accent ?? "text-neutral-100"}`}>{value}</span>
    </div>
  );
}

function LineChart({
  title,
  series,
  height = 150,
}: {
  title: string;
  series: { color: string; label: string; points: number[] }[];
  height?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const c = ref.current;
    const wrap = wrapRef.current;
    if (!c || !wrap) return;
    const w = Math.max(120, wrap.clientWidth);
    c.width = w;
    c.height = height;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, w, height);
    const all = series.flatMap((s) => s.points).filter((v) => Number.isFinite(v));
    if (all.length === 0) {
      ctx.fillStyle = "rgba(255,255,255,0.2)";
      ctx.font = "11px ui-sans-serif";
      ctx.fillText("waiting for data…", 10, height / 2);
      return;
    }
    const maxLen = Math.max(...series.map((s) => s.points.length), 2);
    const lo = Math.min(...all);
    let hi = Math.max(...all);
    if (hi - lo < 1e-6) hi = lo + 1;
    const pad = 6;
    const px = (i: number) => pad + (i / (maxLen - 1)) * (w - 2 * pad);
    const py = (v: number) => height - pad - ((v - lo) / (hi - lo)) * (height - 2 * pad);
    // gridlines
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (let k = 0; k <= 3; k++) {
      const y = pad + (k / 3) * (height - 2 * pad);
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(w - pad, y);
      ctx.stroke();
    }
    for (const s of series) {
      if (s.points.length < 1) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      s.points.forEach((v, i) => {
        const x = px(i);
        const y = py(v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "10px ui-monospace";
    ctx.fillText(hi.toFixed(3), pad, pad + 8);
    ctx.fillText(lo.toFixed(3), pad, height - pad);
  }, [series, height]);

  return (
    <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-semibold text-neutral-300">{title}</span>
        <div className="flex gap-3">
          {series.map((s) => (
            <span key={s.label} className="flex items-center gap-1 text-[10px] text-neutral-400">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      </div>
      <div ref={wrapRef} className="w-full">
        <canvas ref={ref} />
      </div>
    </div>
  );
}

export function MetricsDock() {
  const epochs = useTrain((s) => s.epochs);
  const sps = useTrain((s) => s.samplesPerSec);
  const gradNorms = useTrain((s) => s.gradNorms);
  const config = useTrain((s) => s.config);
  const last: EpochMetric | undefined = epochs[epochs.length - 1];
  const bestVal = epochs.length ? Math.min(...epochs.map((e) => e.val_loss)) : null;
  const hasAcc = last?.val_acc != null;
  const gradMax = Math.max(1e-9, ...Object.values(gradNorms));

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-2 lg:grid-cols-6">
        <Tile label="epoch" value={last ? `${last.epoch + 1}/${config.epochs}` : `0/${config.epochs}`} />
        <Tile label="train loss" value={last ? last.train_loss.toFixed(4) : "—"} accent="text-sky-300" />
        <Tile label="val loss" value={last ? last.val_loss.toFixed(4) : "—"} accent="text-violet-300" />
        <Tile label="val acc" value={hasAcc ? `${(last!.val_acc! * 100).toFixed(1)}%` : "—"} accent="text-emerald-300" />
        <Tile label="lr" value={last ? last.lr.toExponential(1) : "—"} />
        <Tile label="samples/s" value={sps ? sps.toFixed(0) : "—"} />
      </div>

      <LineChart
        title="Loss"
        series={[
          { color: "#38bdf8", label: "train", points: epochs.map((e) => e.train_loss) },
          { color: "#a78bfa", label: "val", points: epochs.map((e) => e.val_loss) },
        ]}
      />
      {hasAcc && (
        <LineChart
          title="Accuracy"
          series={[
            { color: "#34d399", label: "train", points: epochs.map((e) => e.train_acc ?? 0) },
            { color: "#fbbf24", label: "val", points: epochs.map((e) => e.val_acc ?? 0) },
          ]}
        />
      )}

      {Object.keys(gradNorms).length > 0 && (
        <div className="rounded-xl border border-neutral-800 bg-neutral-900/40 p-3">
          <div className="mb-2 text-xs font-semibold text-neutral-300">Gradient flow (‖∇w‖ per layer)</div>
          <div className="space-y-1">
            {Object.entries(gradNorms).map(([nid, v]) => (
              <div key={nid} className="flex items-center gap-2">
                <span className="w-24 shrink-0 truncate font-mono text-[10px] text-neutral-500">{nid}</span>
                <div className="h-2 flex-1 overflow-hidden rounded bg-neutral-800">
                  <div className="h-full rounded bg-sky-500/70" style={{ width: `${(v / gradMax) * 100}%` }} />
                </div>
                <span className="w-16 shrink-0 text-right font-mono text-[10px] text-neutral-400">{v.toExponential(1)}</span>
              </div>
            ))}
          </div>
          {bestVal != null && (
            <div className="mt-2 text-[10px] text-neutral-500">best val loss so far: {bestVal.toFixed(4)}</div>
          )}
        </div>
      )}
    </div>
  );
}
