"use client";

import { useRouter } from "next/navigation";
import { useObserve } from "@/lib/observeStore";

export function LayerInspector() {
  const router = useRouter();
  const snapshot = useObserve((s) => s.snapshot);
  const selected = useObserve((s) => s.selectedNode);
  if (!snapshot || !selected) {
    return <div className="p-4 text-xs text-neutral-500">Select or step to a node to inspect its activations.</div>;
  }
  const rec = snapshot.nodes[selected];
  if (!rec) return <div className="p-4 text-xs text-neutral-500">No data for this node.</div>;
  const sm = rec.summary;
  const isLinear = snapshot.node_types[selected] === "linear";

  return (
    <div className="flex min-h-0 flex-1 flex-col p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-500">Activations</span>
        <span className="font-mono text-xs text-neutral-400">{selected} · [{rec.shape.join(", ")}]</span>
      </div>
      {isLinear && (
        <button
          onClick={() => router.push(`/build/${selected}?mode=live`)}
          className="mb-3 w-full rounded-lg border border-violet-500/50 px-3 py-1.5 text-xs text-violet-200 transition hover:bg-violet-500/10"
        >
          Edit weights (live) ↗ — then re-run to see the effect
        </button>
      )}
      <div className="grid grid-cols-3 gap-1.5 text-[10px]">
        {[["min", sm.min], ["max", sm.max], ["mean", sm.mean], ["std", sm.std], ["‖·‖", sm.l2], ["active", sm.frac_active]].map(([k, v]) => (
          <div key={k as string} className="rounded bg-neutral-800/50 px-2 py-1 ring-1 ring-inset ring-neutral-800">
            <div className="uppercase tracking-wide text-neutral-500">{k}</div>
            <div className="font-mono text-neutral-200">{(v as number).toFixed(3)}</div>
          </div>
        ))}
      </div>

      {rec.values ? (
        <div className="mt-3 min-h-0 flex-1 space-y-1 overflow-auto">
          {rec.values.map((v, i) => {
            const w = Math.min(100, (Math.abs(v) / (sm.absmax || 1)) * 100);
            return (
              <div key={i} className="flex items-center gap-2">
                <span className="w-8 shrink-0 font-mono text-[10px] text-neutral-600">n{i}</span>
                <div className="relative h-3 flex-1">
                  <div className="absolute left-1/2 top-0 h-full w-px bg-neutral-700" />
                  <div
                    className={`absolute top-0 h-full ${v >= 0 ? "left-1/2 bg-sky-500/70" : "right-1/2 bg-amber-500/70"}`}
                    style={{ width: `${w / 2}%` }}
                  />
                </div>
                <span className="w-16 shrink-0 text-right font-mono text-[10px] text-neutral-400">{v.toFixed(3)}</span>
              </div>
            );
          })}
        </div>
      ) : rec.channels ? (
        <div className="mt-3 text-xs text-neutral-400">
          {rec.channels.length} channels · showing feature-map thumbnails is a later addition.
          <div className="mt-2 flex flex-wrap gap-1">
            {(rec.thumbs?.images ?? []).map((img, i) => (
              <ChannelThumb key={i} data={img} hw={rec.thumbs!.hw} />
            ))}
          </div>
        </div>
      ) : (
        <div className="mt-3 text-xs text-neutral-500">{rec.truncated ? "Too large to list inline." : "No per-unit values."}</div>
      )}
    </div>
  );
}

function ChannelThumb({ data, hw }: { data: number[]; hw: number }) {
  const ref = (c: HTMLCanvasElement | null) => {
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(hw, hw);
    for (let i = 0; i < hw * hw; i++) {
      const v = Math.round((data[i] ?? 0) * 255);
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  };
  return <canvas ref={ref} width={hw} height={hw} className="h-8 w-8 rounded border border-neutral-700" />;
}
