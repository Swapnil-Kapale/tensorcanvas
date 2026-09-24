// Pure serialization: canvas nodes/edges → the backend's network spec.

import type { Edge, Node } from "@xyflow/react";
import { CATALOG_BY_TYPE, type FlowNodeData } from "@/components/build/nodeCatalog";
import type { NetworkSpecWire, NodeSpecWire } from "@/lib/api";

/** Coerce a single typed field value to what the backend expects. */
function coerceField(kind: string, raw: string): unknown {
  const v = (raw ?? "").trim();
  if (v === "") return undefined; // omit → backend reports "missing required …"
  if (kind === "int-list") {
    const parts = v.split(",").map((s) => s.trim()).filter((s) => s !== "");
    const nums = parts.map(Number);
    return parts.length > 0 && nums.every((n) => Number.isInteger(n)) ? nums : v;
  }
  // number
  const n = Number(v);
  return Number.isNaN(n) ? v : n;
}

function paramsFor(data: FlowNodeData): Record<string, unknown> {
  const entry = CATALOG_BY_TYPE[data.type];
  const out: Record<string, unknown> = {};
  for (const f of entry?.fields ?? []) {
    const val = coerceField(f.kind, data.params[f.key] ?? "");
    if (val !== undefined) out[f.key] = val;
  }
  if (entry?.activation && data.activation && data.activation !== "none") {
    out.activation = data.activation;
  }
  if (entry?.bias && typeof data.bias === "boolean") {
    out.bias = data.bias;
  }
  if (data.initWeights) out.init_weights = data.initWeights;
  if (data.initBias) out.init_bias = data.initBias;
  return out;
}

export function canvasToSpec(
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  name?: string | null,
): NetworkSpecWire {
  const nodeSpecs: NodeSpecWire[] = nodes.map((n) => ({
    id: n.id,
    type: n.data.type,
    params: paramsFor(n.data),
  }));
  const edgeSpecs = edges.map((e) => ({ source: e.source, target: e.target }));
  return { name: name ?? null, nodes: nodeSpecs, edges: edgeSpecs };
}
