"use client";

import { useEffect, useState } from "react";
import { useGraph } from "@/lib/store";
import { useObserve } from "@/lib/observeStore";
import { fetchModelIO, type ModelIO } from "@/lib/trainApi";
import { NetworkCanvas } from "./NetworkCanvas";
import { InputPicker } from "./InputPicker";
import { ForwardPassPlayer } from "./ForwardPassPlayer";
import { LayerInspector } from "./LayerInspector";
import { OutputPanel } from "./OutputPanel";
import { SnapshotBrowser } from "./SnapshotBrowser";

export function ObserveWorkspace() {
  const modelId = useGraph((s) => s.lastModelId);
  const refreshList = useObserve((s) => s.refreshList);
  const clear = useObserve((s) => s.clear);
  const play = useObserve((s) => s.play);
  const pause = useObserve((s) => s.pause);
  const stepFn = useObserve((s) => s.step);
  const error = useObserve((s) => s.error);
  const [modelIO, setModelIO] = useState<ModelIO | null>(null);

  useEffect(() => {
    if (!modelId) return;
    fetchModelIO(modelId).then(setModelIO).catch(() => setModelIO(null));
    refreshList(modelId);
    return () => clear();
  }, [modelId, refreshList, clear]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && ["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName)) return;
      if (e.key === " ") {
        e.preventDefault();
        if (useObserve.getState().playing) pause();
        else play();
      } else if (e.key === "ArrowLeft") {
        stepFn(-1);
      } else if (e.key === "ArrowRight") {
        stepFn(1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [play, pause, stepFn]);

  if (!modelId) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-[var(--faint)]">
        <div className="text-base font-medium text-[var(--muted)]">No compiled model yet</div>
        <p className="max-w-sm text-sm">Build and Compile a network first, then run it here to watch the forward pass.</p>
      </div>
    );
  }

  return (
    <div className="relative h-full min-h-0">
      {/* base canvas */}
      <div className="tc-stage absolute inset-0">
        <NetworkCanvas />
      </div>

      {/* floating input + snapshots (left) */}
      <aside className="tc-panel absolute bottom-4 left-4 top-4 z-20 flex w-80 flex-col overflow-auto">
        <InputPicker modelId={modelId} inputShape={modelIO?.input_shape ?? [2]} hasConv={modelIO?.has_conv ?? false} />
        <SnapshotBrowser modelId={modelId} />
      </aside>

      {/* floating output + layer inspector (right) */}
      <aside className="tc-panel absolute bottom-4 right-4 top-4 z-20 flex w-80 flex-col overflow-auto">
        <OutputPanel />
        <LayerInspector />
      </aside>

      {/* floating transport + error */}
      <div className="absolute bottom-4 left-1/2 z-20 -translate-x-1/2">
        {error && (
          <div className="tc-glass mb-2 rounded-xl border border-red-500/30 px-4 py-1.5 text-center text-xs text-red-300">{error}</div>
        )}
        <ForwardPassPlayer />
      </div>
    </div>
  );
}
