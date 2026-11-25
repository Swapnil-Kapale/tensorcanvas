"use client";

import { useObserve } from "@/lib/observeStore";

export function OutputPanel() {
  const snapshot = useObserve((s) => s.snapshot);
  if (!snapshot) return null;
  const pred = snapshot.output.prediction;
  const target = pred.target;

  if (pred.task === "regression") {
    return (
      <div className="border-b border-[var(--stroke)] p-4">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--faint)]">Output</div>
        <div className="font-mono text-sm text-[var(--text)]">{(pred.scalar ?? []).map((v) => v.toFixed(4)).join(", ")}</div>
        {target != null && <div className="mt-1 text-xs text-[var(--faint)]">target: {String(target)}</div>}
      </div>
    );
  }

  const probs = pred.probs ?? [];
  const correct = target != null && target === pred.pred_class;
  // Prefer a labelled top-k (e.g. pretrained ImageNet) or cap long prob lists.
  const rows =
    pred.topk && pred.topk.length
      ? pred.topk.map((t) => ({ i: t.index, p: t.prob, label: t.label }))
      : probs
          .map((p, i) => ({ i, p, label: null as string | null }))
          .sort((a, b) => b.p - a.p)
          .slice(0, 12);
  const max = Math.max(...rows.map((r) => r.p), 1e-9);

  return (
    <div className="border-b border-[var(--stroke)] p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--faint)]">Output</span>
        {target != null && (
          <span className={`rounded px-1.5 py-0.5 text-[10px] ${correct ? "bg-emerald-500/15 text-emerald-300" : "bg-red-500/15 text-red-300"}`}>
            {correct ? "✓ correct" : `✗ pred ${pred.pred_class} / true ${target}`}
          </span>
        )}
      </div>
      <div className="space-y-1.5">
        {rows.map((r) => (
          <div key={r.i} className="flex items-center gap-2">
            <span
              className={`w-24 shrink-0 truncate font-mono text-[11px] ${r.i === pred.pred_class ? "text-emerald-300" : "text-[var(--faint)]"}`}
              title={r.label ?? `class ${r.i}`}
            >
              {r.label ?? `c${r.i}`}
            </span>
            <div className="h-3 flex-1 overflow-hidden rounded bg-black/30">
              <div
                className={`h-full rounded ${r.i === pred.pred_class ? "bg-emerald-500/80" : "bg-[var(--stroke-strong)]"}`}
                style={{ width: `${(r.p / max) * 100}%` }}
              />
            </div>
            <span className="w-14 shrink-0 text-right font-mono text-[11px] text-[var(--muted)]">{(r.p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
