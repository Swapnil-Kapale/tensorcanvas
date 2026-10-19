"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useGraph } from "@/lib/store";
import { useTrain } from "@/lib/trainStore";
import {
  controlRun,
  fetchModelIO,
  openRunStream,
  restoreCheckpoint,
  startTraining,
  validateDataset,
  type DatasetInfo,
  type ModelIO,
} from "@/lib/trainApi";
import { DatasetPanel } from "./DatasetPanel";
import { ConfigPanel } from "./ConfigPanel";
import { MetricsDock } from "./MetricsDock";
import { DecisionBoundary } from "./DecisionBoundary";

const STATUS_STYLE: Record<string, string> = {
  idle: "text-neutral-400",
  running: "text-sky-300",
  paused: "text-amber-300",
  completed: "text-emerald-300",
  early_stopped: "text-emerald-300",
  stopped: "text-neutral-300",
  error: "text-red-300",
};

export function TrainWorkspace() {
  const modelId = useGraph((s) => s.lastModelId);
  const { dataset, config, runId, status, boundary, checkpoints, error, setDataset, beginRun, onSSE } = useTrain();

  const [modelIO, setModelIO] = useState<ModelIO | null>(null);
  const [validation, setValidation] = useState<{ forId: string; ok: boolean; detail?: string } | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!modelId) return;
    fetchModelIO(modelId).then(setModelIO).catch(() => setModelIO(null));
  }, [modelId]);

  useEffect(() => {
    return () => esRef.current?.close();
  }, []);

  // Validate whenever the dataset changes (result keyed to the dataset id).
  useEffect(() => {
    if (!modelId || !dataset) return;
    let cancelled = false;
    validateDataset(modelId, dataset.dataset_id).then((r) => {
      if (!cancelled) setValidation({ forId: dataset.dataset_id, ok: Boolean(r.ok), detail: r.detail });
    });
    return () => {
      cancelled = true;
    };
  }, [modelId, dataset]);

  const val = validation && dataset && validation.forId === dataset.dataset_id ? validation : null;

  const onDataset = useCallback((d: DatasetInfo) => setDataset(d), [setDataset]);

  const start = useCallback(async () => {
    if (!modelId || !dataset) return;
    try {
      const { run_id } = await startTraining(modelId, { dataset_id: dataset.dataset_id, ...config });
      beginRun(run_id);
      esRef.current?.close();
      esRef.current = openRunStream(run_id, onSSE);
    } catch (e) {
      onSSE("error", { message: e instanceof Error ? e.message : String(e) });
    }
  }, [modelId, dataset, config, beginRun, onSSE]);

  const control = useCallback(
    (a: "pause" | "resume" | "stop") => {
      if (runId) controlRun(runId, a).catch(() => {});
    },
    [runId],
  );

  const scrub = useCallback(
    (ckptId: string) => {
      if (runId) restoreCheckpoint(runId, ckptId).catch(() => {});
    },
    [runId],
  );

  if (!modelId) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-neutral-500">
        <div className="text-base font-medium text-neutral-300">No compiled model yet</div>
        <p className="max-w-sm text-sm">Go to the Build tab, draw a network and press Compile — then come back here to train it.</p>
      </div>
    );
  }

  const running = status === "running";
  const paused = status === "paused";
  const active = running || paused;
  const twoD = (modelIO?.input_shape?.length === 1 && modelIO.input_shape[0] === 2) || Boolean(boundary);
  const scatter = dataset?.scatter ?? null;

  return (
    <div className="flex h-full min-h-0">
      {/* left: dataset + config */}
      <aside className="flex w-[340px] shrink-0 flex-col overflow-y-auto border-r border-neutral-800 bg-neutral-900/30">
        <DatasetPanel modelIO={modelIO} dataset={dataset} validation={val} disabled={active} onDataset={onDataset} />
        <ConfigPanel disabled={active} />
      </aside>

      {/* main */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {/* transport bar */}
        <div className="flex flex-wrap items-center gap-2 border-b border-neutral-800 px-5 py-3">
          <button
            onClick={start}
            disabled={!dataset || !val?.ok || running}
            className="h-9 rounded-lg bg-sky-500 px-4 text-sm font-semibold text-neutral-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-40"
            title={!val?.ok ? "Pick a dataset that matches this network" : "Start training"}
          >
            {active ? "Restart" : "Start training"}
          </button>
          {running && (
            <button onClick={() => control("pause")} className="h-9 rounded-lg border border-neutral-700 px-3 text-sm text-neutral-300 hover:bg-neutral-800">
              Pause
            </button>
          )}
          {paused && (
            <button onClick={() => control("resume")} className="h-9 rounded-lg border border-neutral-700 px-3 text-sm text-neutral-300 hover:bg-neutral-800">
              Resume
            </button>
          )}
          {active && (
            <button onClick={() => control("stop")} className="h-9 rounded-lg border border-red-500/40 px-3 text-sm text-red-300 hover:bg-red-500/10">
              Stop
            </button>
          )}
          <span className={`ml-2 text-sm font-medium ${STATUS_STYLE[status] ?? "text-neutral-400"}`}>● {status}</span>
          {error && <span className="text-xs text-red-300">{error}</span>}

          {/* checkpoint rewind */}
          {checkpoints.length > 1 && !active && (
            <div className="ml-auto flex items-center gap-2">
              <span className="text-[11px] text-neutral-500">rewind</span>
              <input
                type="range"
                min={0}
                max={checkpoints.length - 1}
                defaultValue={checkpoints.length - 1}
                onChange={(e) => scrub(checkpoints[Number(e.target.value)].ckpt_id)}
                className="w-40 accent-sky-500"
              />
              <span className="font-mono text-[11px] text-neutral-500">{checkpoints.length} epochs</span>
            </div>
          )}
        </div>

        {/* metrics + boundary */}
        <div className="min-h-0 flex-1 overflow-y-auto p-5">
          <div className="flex flex-col gap-5 xl:flex-row">
            {twoD && (
              <div className="shrink-0">
                <div className="mb-2 text-xs font-semibold text-neutral-300">
                  Decision boundary{boundary ? ` · epoch ${boundary.epoch + 1}` : ""}
                </div>
                <DecisionBoundary boundary={boundary} scatter={scatter} size={380} />
              </div>
            )}
            <div className="min-w-0 flex-1">
              <MetricsDock />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
