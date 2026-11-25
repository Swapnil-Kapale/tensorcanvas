"use client";

import { useObserve } from "@/lib/observeStore";

export function SnapshotBrowser({ modelId }: { modelId: string }) {
  const snapshots = useObserve((s) => s.snapshots);
  const loadSnapshot = useObserve((s) => s.loadSnapshot);
  const remove = useObserve((s) => s.remove);
  const current = useObserve((s) => s.snapshot?.snapshot_id);

  return (
    <div className="flex min-h-0 flex-1 flex-col border-t border-[var(--stroke)] p-4">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--faint)]">
        Snapshots · {snapshots.length}
      </div>
      {snapshots.length === 0 && <div className="text-[11px] text-[var(--faint)]">Runs are saved here for analysis.</div>}
      <div className="min-h-0 flex-1 space-y-1 overflow-auto">
        {snapshots.map((s) => (
          <div
            key={s.snapshot_id}
            className={[
              "flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs",
              s.snapshot_id === current ? "bg-violet-500/10 ring-1 ring-inset ring-violet-500/30" : "hover:bg-[var(--glass-hover)]",
            ].join(" ")}
          >
            <button onClick={() => loadSnapshot(s.snapshot_id)} className="flex min-w-0 flex-1 items-center gap-2 text-left">
              <span className="font-mono text-[var(--text)]">
                {s.pred_class != null ? `class ${s.pred_class}` : "run"}
              </span>
              {s.confidence != null && <span className="text-[var(--faint)]">{(s.confidence * 100).toFixed(0)}%</span>}
              <span className="ml-auto truncate text-[10px] text-[var(--faint)]">{new Date(s.created_at).toLocaleTimeString()}</span>
            </button>
            {s.stale && <span title="weights changed since this run" className="h-2 w-2 shrink-0 rounded-full bg-amber-400" />}
            <button onClick={() => remove(s.snapshot_id, modelId)} className="shrink-0 text-[var(--faint)] hover:text-red-400" aria-label="delete">
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
