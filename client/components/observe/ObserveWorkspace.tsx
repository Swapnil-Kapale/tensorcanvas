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
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-neutral-500">
        <div className="text-base font-medium text-neutral-300">No compiled model yet</div>
        <p className="max-w-sm text-sm">Build and Compile a network first, then run it here to watch the forward pass.</p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0">
      <aside className="flex w-80 shrink-0 flex-col overflow-hidden border-r border-neutral-800 bg-neutral-900/30">
        <InputPicker modelId={modelId} inputShape={modelIO?.input_shape ?? [2]} />
        <SnapshotBrowser modelId={modelId} />
      </aside>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1">
          <NetworkCanvas />
        </div>
        {error && <div className="border-t border-red-500/30 bg-red-500/10 px-4 py-1.5 text-xs text-red-300">{error}</div>}
        <ForwardPassPlayer />
      </div>

      <aside className="flex w-80 shrink-0 flex-col overflow-hidden border-l border-neutral-800 bg-neutral-900/30">
        <OutputPanel />
        <LayerInspector />
      </aside>
    </div>
  );
}
