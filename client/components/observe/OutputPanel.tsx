"use client";

import { useObserve } from "@/lib/observeStore";

export function OutputPanel() {
  const snapshot = useObserve((s) => s.snapshot);
  if (!snapshot) return null;
  const pred = snapshot.output.prediction;
  const target = pred.target;

  if (pred.task === "regression") {
    return (
      <div className="border-b border-neutral-800 p-4">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-neutral-500">Output</div>
        <div className="font-mono text-sm text-neutral-100">{(pred.scalar ?? []).map((v) => v.toFixed(4)).join(", ")}</div>
        {target != null && <div className="mt-1 text-xs text-neutral-500">target: {String(target)}</div>}
      </div>
    );
  }

  const probs = pred.probs ?? [];
  const max = Math.max(...probs, 1e-9);
  const correct = target != null && target === pred.pred_class;

  return (
    <div className="border-b border-neutral-800 p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-500">Output</span>
        {target != null && (
          <span className={`rounded px-1.5 py-0.5 text-[10px] ${correct ? "bg-emerald-500/15 text-emerald-300" : "bg-red-500/15 text-red-300"}`}>
            {correct ? "✓ correct" : `✗ pred ${pred.pred_class} / true ${target}`}
          </span>
        )}
      </div>
      <div className="space-y-1.5">
        {probs.map((p, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className={`w-10 shrink-0 font-mono text-[11px] ${i === pred.pred_class ? "text-emerald-300" : "text-neutral-500"}`}>
              c{i}
            </span>
            <div className="h-3 flex-1 overflow-hidden rounded bg-neutral-800">
              <div
                className={`h-full rounded ${i === pred.pred_class ? "bg-emerald-500/80" : "bg-neutral-600"}`}
                style={{ width: `${(p / max) * 100}%` }}
              />
            </div>
            <span className="w-14 shrink-0 text-right font-mono text-[11px] text-neutral-400">{(p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
