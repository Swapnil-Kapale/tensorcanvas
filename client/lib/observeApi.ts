// Client for the Observe/run API.

import { API_BASE } from "@/lib/api";

export type RunInput =
  | { kind: "vector"; values: number[] }
  | { kind: "dataset"; dataset_id: string; index: number }
  | { kind: "random"; seed: number };

export interface NodeRec {
  shape: number[];
  summary: { min: number; max: number; mean: number; std: number; absmax: number; l2: number; frac_active: number };
  mag?: number[] | null;
  values?: number[] | null;
  channels?: { c: number; mean: number; absmax: number }[];
  thumbs?: { hw: number; images: number[][] };
  truncated?: boolean;
}

export interface Prediction {
  task: "classification" | "regression";
  pred_class?: number;
  probs?: number[];
  confidence?: number;
  scalar?: number[];
  target?: number | number[] | null;
}

export interface ObserveSnapshot {
  snapshot_id?: string;
  model_id?: string;
  created_at?: string;
  label?: string | null;
  stale?: boolean;
  weight_version: number;
  order: string[];
  node_types: Record<string, string>;
  input: { source: Record<string, unknown>; shape: number[]; values: number[] | null; thumb: number[] | null };
  nodes: Record<string, NodeRec>;
  output: { shape: number[]; values: number[] | null; prediction: Prediction };
}

export interface SnapshotMeta {
  snapshot_id: string;
  created_at: string;
  weight_version: number;
  label: string | null;
  input_source: Record<string, unknown>;
  pred_class: number | null;
  confidence: number | null;
  stale: boolean;
}

async function jsonOrThrow(res: Response) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail ?? `HTTP ${res.status}`);
  return body;
}

export async function runModel(
  modelId: string,
  input: RunInput,
  opts: { save?: boolean; label?: string } = {},
): Promise<ObserveSnapshot> {
  return jsonOrThrow(
    await fetch(`${API_BASE}/api/v1/models/${modelId}/run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ input, save: opts.save ?? true, label: opts.label }),
    }),
  );
}

export async function listSnapshots(modelId: string): Promise<SnapshotMeta[]> {
  const b = await jsonOrThrow(await fetch(`${API_BASE}/api/v1/models/${modelId}/snapshots`));
  return b.snapshots as SnapshotMeta[];
}

export async function getSnapshot(id: string): Promise<ObserveSnapshot> {
  return jsonOrThrow(await fetch(`${API_BASE}/api/v1/snapshots/${id}`));
}

export async function deleteSnapshot(id: string) {
  return jsonOrThrow(await fetch(`${API_BASE}/api/v1/snapshots/${id}`, { method: "DELETE" }));
}

export async function patchNodeWeights(
  modelId: string,
  nodeId: string,
  edit: { weights?: number[][]; bias?: number[] | null; cells?: { i: number; j: number; v: number }[] },
): Promise<{ ok: boolean; weight_version: number }> {
  return jsonOrThrow(
    await fetch(`${API_BASE}/api/v1/models/${modelId}/nodes/${nodeId}/weights`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(edit),
    }),
  );
}
