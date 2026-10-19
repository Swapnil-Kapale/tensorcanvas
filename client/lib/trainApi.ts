// Client for the training API (uploads, run control, SSE stream).

import { API_BASE } from "@/lib/api";

export interface DatasetInfo {
  dataset_id: string;
  format: string;
  kind: string;
  task: string;
  n_samples: number;
  n_classes: number | null;
  feature_shape: number[] | null;
  scatter?: { points: number[][]; labels: number[] | null };
  thumbs?: { size: number; images: number[][]; labels: number[] };
}

export interface ModelIO {
  input_shape: number[];
  out_dim: number;
  builtins: { id: string; blurb: string }[];
}

export interface TrainConfig {
  dataset_id: string;
  val_split: number;
  batch_size: number;
  shuffle: boolean;
  standardize: boolean;
  loss: string;
  label_smoothing: number;
  optimizer: string;
  lr: number;
  momentum: number;
  weight_decay: number;
  nesterov: boolean;
  beta1: number;
  beta2: number;
  eps: number;
  scheduler: string;
  step_size: number;
  gamma: number;
  t_max: number;
  sched_patience: number;
  factor: number;
  min_lr: number;
  warmup_epochs: number;
  epochs: number;
  grad_accum: number;
  grad_clip: number | null;
  early_stopping: { enabled: boolean; monitor: string; patience: number; min_delta: number; restore_best: boolean };
  init: string;
  seed: number;
  deterministic: boolean;
  log_every: number;
  frame_every: number;
}

export const DEFAULT_CONFIG: Omit<TrainConfig, "dataset_id"> = {
  val_split: 0.2,
  batch_size: 32,
  shuffle: true,
  standardize: false,
  loss: "auto",
  label_smoothing: 0,
  optimizer: "adam",
  lr: 0.01,
  momentum: 0.9,
  weight_decay: 0,
  nesterov: false,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  scheduler: "none",
  step_size: 30,
  gamma: 0.1,
  t_max: 50,
  sched_patience: 5,
  factor: 0.1,
  min_lr: 0,
  warmup_epochs: 0,
  epochs: 100,
  grad_accum: 1,
  grad_clip: null,
  early_stopping: { enabled: false, monitor: "val_loss", patience: 10, min_delta: 0, restore_best: true },
  init: "keep",
  seed: 42,
  deterministic: false,
  log_every: 10,
  frame_every: 2,
};

async function jsonOrThrow(res: Response) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail ?? `HTTP ${res.status}`);
  return body;
}

export async function fetchModelIO(modelId: string): Promise<ModelIO> {
  return jsonOrThrow(await fetch(`${API_BASE}/api/v1/models/${modelId}/datasets`));
}

export async function uploadDataset(file: File): Promise<DatasetInfo> {
  const fd = new FormData();
  fd.append("file", file);
  return jsonOrThrow(await fetch(`${API_BASE}/api/v1/datasets`, { method: "POST", body: fd }));
}

export async function builtinDataset(name: string, params: Record<string, number> = {}): Promise<DatasetInfo> {
  const q = new URLSearchParams({ name, ...Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)])) });
  return jsonOrThrow(await fetch(`${API_BASE}/api/v1/datasets/builtin?${q}`, { method: "POST" }));
}

export async function validateDataset(modelId: string, datasetId: string) {
  const res = await fetch(`${API_BASE}/api/v1/models/${modelId}/datasets/validate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId }),
  });
  const body = await res.json().catch(() => ({}));
  return { ok: res.ok && body.ok, detail: body.detail as string | undefined, ...body };
}

export async function startTraining(modelId: string, cfg: TrainConfig): Promise<{ run_id: string }> {
  return jsonOrThrow(
    await fetch(`${API_BASE}/api/v1/models/${modelId}/train`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(cfg),
    }),
  );
}

export async function controlRun(runId: string, action: "pause" | "resume" | "stop") {
  return jsonOrThrow(await fetch(`${API_BASE}/api/v1/runs/${runId}/${action}`, { method: "POST" }));
}

export async function fetchCheckpoints(runId: string) {
  return jsonOrThrow(await fetch(`${API_BASE}/api/v1/runs/${runId}/checkpoints`));
}

export async function restoreCheckpoint(runId: string, ckptId: string) {
  return jsonOrThrow(
    await fetch(`${API_BASE}/api/v1/runs/${runId}/checkpoints/${ckptId}/restore`, { method: "POST" }),
  );
}

const SSE_EVENTS = ["hello", "metrics", "epoch", "frame", "status", "done", "error"] as const;

/** Open an EventSource for a run; calls onEvent(type, data) per frame. Returns the source. */
export function openRunStream(runId: string, onEvent: (type: string, data: Record<string, unknown>) => void): EventSource {
  const es = new EventSource(`${API_BASE}/api/v1/runs/${runId}/stream`);
  for (const t of SSE_EVENTS) {
    es.addEventListener(t, (e) => {
      try {
        onEvent(t, JSON.parse((e as MessageEvent).data));
      } catch {
        /* ignore malformed frame */
      }
      if (t === "done" || t === "error") es.close();
    });
  }
  return es;
}
