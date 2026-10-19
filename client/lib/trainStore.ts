"use client";

import { create } from "zustand";
import { useGraph } from "@/lib/store";
import { DEFAULT_CONFIG, type DatasetInfo, type TrainConfig } from "@/lib/trainApi";

const MAX_STEP_POINTS = 2000;

export interface EpochMetric {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_acc: number | null;
  val_acc: number | null;
  lr: number;
}

export interface Boundary {
  res: number;
  mode: "prob" | "class";
  n_classes?: number;
  bounds: { xmin: number; xmax: number; ymin: number; ymax: number };
  values: number[];
  epoch: number;
}

export interface Checkpoint {
  ckpt_id: string;
  epoch: number;
  val_loss: number;
  val_acc: number | null;
  is_best: boolean;
}

export type RunStatus = "idle" | "running" | "paused" | "completed" | "stopped" | "error" | "early_stopped";

interface TrainState {
  dataset: DatasetInfo | null;
  config: Omit<TrainConfig, "dataset_id">;
  runId: string | null;
  status: RunStatus;
  hello: Record<string, unknown> | null;
  stepLoss: { step: number; loss: number }[];
  epochs: EpochMetric[];
  gradNorms: Record<string, number>;
  boundary: Boundary | null;
  checkpoints: Checkpoint[];
  error: string | null;
  samplesPerSec: number;

  setDataset: (d: DatasetInfo | null) => void;
  setConfig: (patch: Partial<Omit<TrainConfig, "dataset_id">>) => void;
  beginRun: (runId: string) => void;
  onSSE: (type: string, data: Record<string, unknown>) => void;
  reset: () => void;
}

const initial = {
  dataset: null as DatasetInfo | null,
  config: { ...DEFAULT_CONFIG },
  runId: null as string | null,
  status: "idle" as RunStatus,
  hello: null as Record<string, unknown> | null,
  stepLoss: [] as { step: number; loss: number }[],
  epochs: [] as EpochMetric[],
  gradNorms: {} as Record<string, number>,
  boundary: null as Boundary | null,
  checkpoints: [] as Checkpoint[],
  error: null as string | null,
  samplesPerSec: 0,
};

export const useTrain = create<TrainState>((set) => ({
  ...initial,

  setDataset: (d) => set({ dataset: d }),
  setConfig: (patch) => set((s) => ({ config: { ...s.config, ...patch } })),

  beginRun: (runId) =>
    set({
      runId,
      status: "running",
      error: null,
      hello: null,
      stepLoss: [],
      epochs: [],
      gradNorms: {},
      boundary: null,
      checkpoints: [],
    }),

  onSSE: (type, data) =>
    set((s) => {
      switch (type) {
        case "hello":
          return { hello: data, status: "running" };
        case "metrics": {
          const next = [...s.stepLoss, { step: data.global_step as number, loss: data.loss as number }];
          if (next.length > MAX_STEP_POINTS) next.splice(0, next.length - MAX_STEP_POINTS);
          return { stepLoss: next, samplesPerSec: (data.samples_per_sec as number) ?? s.samplesPerSec };
        }
        case "epoch": {
          const em: EpochMetric = {
            epoch: data.epoch as number,
            train_loss: data.train_loss as number,
            val_loss: data.val_loss as number,
            train_acc: (data.train_acc as number) ?? null,
            val_acc: (data.val_acc as number) ?? null,
            lr: data.lr as number,
          };
          // Push trained weights + neuron glow into the graph store (for the
          // heatmap + canvas), without touching undo history.
          const g = useGraph.getState();
          const weights = (data.weights as Record<string, number[][]>) ?? {};
          const mags = (data.neuron_mags as Record<string, number[]>) ?? {};
          const ids = new Set([...Object.keys(weights), ...Object.keys(mags)]);
          if (ids.size) {
            useGraph.setState({
              nodes: g.nodes.map((n) =>
                ids.has(n.id)
                  ? { ...n, data: { ...n.data, weights: weights[n.id] ?? n.data.weights, activationMag: mags[n.id] ?? n.data.activationMag } }
                  : n,
              ),
            });
          }
          return {
            epochs: [...s.epochs, em],
            gradNorms: (data.grad_norms as Record<string, number>) ?? s.gradNorms,
            checkpoints: [
              ...s.checkpoints,
              {
                ckpt_id: data.ckpt_id as string,
                epoch: data.epoch as number,
                val_loss: data.val_loss as number,
                val_acc: (data.val_acc as number) ?? null,
                is_best: false,
              },
            ],
          };
        }
        case "frame":
          if (data.kind === "decision_boundary")
            return {
              boundary: {
                res: data.res as number,
                mode: data.mode as "prob" | "class",
                n_classes: data.n_classes as number | undefined,
                bounds: data.bounds as Boundary["bounds"],
                values: data.values as number[],
                epoch: data.epoch as number,
              },
            };
          return {};
        case "status":
          return { status: (data.status as RunStatus) ?? s.status };
        case "done":
          return { status: (data.status as RunStatus) ?? "completed" };
        case "error":
          return { status: "error", error: (data.message as string) ?? "training error" };
        default:
          return {};
      }
    }),

  reset: () => set({ ...initial, config: { ...DEFAULT_CONFIG } }),
}));
