// Pure serialization: canvas nodes/edges → the backend's network spec.

import { CATALOG_BY_TYPE, type FlowNodeData, type ParamRow } from "@/components/build/nodeCatalog";
import type { NetworkSpecWire, NodeSpecWire } from "@/lib/api";
import type { TCNode } from "@/lib/store";

function coerceField(kind: string, raw: string): unknown {
  const v = (raw ?? "").trim();
  if (v === "") return undefined;
  if (kind === "int-list") {
    const parts = v.split(",").map((s) => s.trim()).filter((s) => s !== "");
    const nums = parts.map(Number);
    return parts.length > 0 && nums.every((n) => Number.isInteger(n)) ? nums : v;
  }
  const n = Number(v);
  return Number.isNaN(n) ? v : n;
}

/** Best-effort coercion for free-form Custom-node param values. */
function coerceLoose(raw: string): unknown {
  const v = (raw ?? "").trim();
  if (v === "") return undefined;
  if (/^-?\d+$/.test(v)) return parseInt(v, 10);
  if (/^-?\d*\.\d+$/.test(v)) return Number(v);
  if (v === "true") return true;
  if (v === "false") return false;
  if (v[0] === "[" || v[0] === "{") {
    try {
      return JSON.parse(v);
    } catch {
      /* fall through */
    }
  }
  return v;
}

function paramsFor(data: FlowNodeData): Record<string, unknown> {
  const entry = CATALOG_BY_TYPE[data.type];
  const out: Record<string, unknown> = {};
  for (const f of entry?.fields ?? []) {
    const val = coerceField(f.kind, data.params[f.key] ?? "");
    if (val !== undefined) out[f.key] = val;
  }
  if (entry?.custom) {
    for (const row of (data.extra ?? []) as ParamRow[]) {
      if (!row.key.trim()) continue;
      const val = coerceLoose(row.value);
      if (val !== undefined) out[row.key.trim()] = val;
    }
  }
  if (entry?.activation && data.activation && data.activation !== "none") {
    out.activation = data.activation;
  }
  if (entry?.bias && typeof data.bias === "boolean") out.bias = data.bias;
  if (data.initWeights) out.init_weights = data.initWeights;
  if (data.initBias) out.init_bias = data.initBias;
  if (data.neuronActivations && data.neuronActivations.some((a) => a)) {
    out.neuron_activations = data.neuronActivations;
  }
  return out;
}

function backendType(data: FlowNodeData): string {
  return data.type === "custom" ? (data.customType ?? "").trim() : data.type;
}

export function canvasToSpec(
  nodes: TCNode[],
  edges: { source: string; target: string }[],
  name?: string | null,
): NetworkSpecWire {
  const nodeSpecs: NodeSpecWire[] = nodes.map((n) => ({
    id: n.id,
    type: backendType(n.data),
    params: paramsFor(n.data),
  }));
  const edgeSpecs = edges.map((e) => ({ source: e.source, target: e.target }));
  return { name: name ?? null, nodes: nodeSpecs, edges: edgeSpecs };
}
