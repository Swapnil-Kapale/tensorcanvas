"use client";

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import {
  ACTIVATION_GLYPH,
  isExpanded,
  neuronCount,
  type FlowNodeData,
} from "./nodeCatalog";
import { NeuronSphere } from "./NeuronSphere";
import { useGraph } from "@/lib/store";

type TCNode = Node<FlowNodeData, "tcNode">;

// Geometry for the expanded (per-neuron) layout — handles align to sphere centres.
const PAD_TOP = 8;
const HEADER = 24;
const SPHERE = 28;
const STRIDE = 36;
const COL_TOP = PAD_TOP + HEADER;

function neuronMagnitudes(weights: number[][] | null | undefined): number[] | null {
  if (!weights || weights.length === 0) return null;
  const means = weights.map((row) =>
    row.length ? row.reduce((a, b) => a + Math.abs(b), 0) / row.length : 0,
  );
  const max = Math.max(...means, 1e-9);
  return means.map((m) => Math.min(1, m / max));
}

function Glyph({ data }: { data: FlowNodeData }) {
  const g = data.activation ? ACTIVATION_GLYPH[data.activation] : "";
  if (!g) return null;
  return (
    <span className="rounded bg-violet-500/25 px-1 text-[10px] font-semibold text-violet-200">
      {g}
    </span>
  );
}

function Header({ data }: { data: FlowNodeData }) {
  return (
    <div className="flex items-center justify-between gap-2" style={{ height: HEADER }}>
      <span className="text-sm font-semibold">{data.label}</span>
      <div className="flex items-center gap-1">
        <Glyph data={data} />
        {data.bias && <span title="bias" className="h-2 w-2 rounded-full bg-amber-400" />}
      </div>
    </div>
  );
}

function Badges({ data }: { data: FlowNodeData }) {
  return (
    <>
      {data.outputShape && !data.error && (
        <div className="mt-1 inline-block rounded bg-emerald-500/15 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
          → [{data.outputShape.join(", ")}]
        </div>
      )}
      {data.error && (
        <div className="mt-1 max-w-[220px] text-[11px] leading-snug text-red-300">{data.error}</div>
      )}
    </>
  );
}

function summary(data: FlowNodeData): string | null {
  if (data.type === "conv2d") {
    const { out_channels, kernel_size } = data.params;
    return `${out_channels ?? "—"} × ${kernel_size ?? "—"}×${kernel_size ?? "—"}`;
  }
  if (data.type === "maxpool2d") return `k=${data.params.kernel_size ?? "—"}`;
  if (data.type === "input") return `[${(data.params.shape ?? "").trim()}]`;
  if (data.type === "custom") return data.customType?.trim() || "(set type)";
  return null;
}

export function FlowNode({ data, selected }: NodeProps<TCNode>) {
  const threshold = useGraph((s) => s.renderThreshold);
  const hasError = Boolean(data.error);
  const showTarget = data.type !== "input";
  const showSource = data.type !== "output";

  const shellCls = [
    "relative rounded-lg border-2 bg-neutral-900/95 px-3 py-2 text-neutral-100 shadow-lg",
    "min-w-[140px] transition-colors",
    hasError ? "border-red-500" : "border-neutral-600",
    selected ? "ring-2 ring-white/60" : "",
  ].join(" ");

  // Expanded Dense: one sphere per neuron, each with its own handles.
  if (data.type === "linear" && isExpanded(data, threshold)) {
    const n = neuronCount(data) ?? 0;
    const mags = neuronMagnitudes(data.weights);
    return (
      <div className={shellCls} style={{ height: COL_TOP + n * STRIDE + 24 }}>
        <Header data={data} />
        <div className="relative" style={{ height: n * STRIDE }}>
          {Array.from({ length: n }).map((_, i) => (
            <div
              key={i}
              className="absolute left-1/2 -translate-x-1/2"
              style={{ top: i * STRIDE + (STRIDE - SPHERE) / 2 }}
            >
              <NeuronSphere size={SPHERE} mag={mags?.[i] ?? null} />
            </div>
          ))}
        </div>
        <Badges data={data} />
        {Array.from({ length: n }).map((_, i) => (
          <Handle
            key={`t${i}`}
            id={`t${i}`}
            type="target"
            position={Position.Left}
            style={{ top: COL_TOP + i * STRIDE + SPHERE / 2 }}
            className="!h-2.5 !w-2.5 !bg-neutral-400"
          />
        ))}
        {Array.from({ length: n }).map((_, i) => (
          <Handle
            key={`s${i}`}
            id={`s${i}`}
            type="source"
            position={Position.Right}
            style={{ top: COL_TOP + i * STRIDE + SPHERE / 2 }}
            className="!h-2.5 !w-2.5 !bg-neutral-400"
          />
        ))}
      </div>
    );
  }

  // Collapsed Dense: sample spheres + count.
  const collapsedDense = data.type === "linear";
  const s = summary(data);
  const mags = neuronMagnitudes(data.weights);
  return (
    <div className={shellCls}>
      {showTarget && (
        <Handle id="tc" type="target" position={Position.Left} className="!h-2.5 !w-2.5 !bg-neutral-400" />
      )}
      <Header data={data} />
      {collapsedDense ? (
        <div className="mt-2 flex flex-col items-center gap-1">
          <div className="flex gap-1.5">
            {[0, 1, 2].map((i) => (
              <NeuronSphere key={i} size={16} mag={mags?.[i] ?? null} />
            ))}
          </div>
          <div className="text-center leading-tight">
            <div className="text-lg font-bold">{neuronCount(data) ?? "—"}</div>
            <div className="text-[10px] text-neutral-400">neurons</div>
          </div>
        </div>
      ) : (
        s && <div className="mt-1 font-mono text-[11px] text-neutral-400">{s}</div>
      )}
      <Badges data={data} />
      {showSource && (
        <Handle id="sc" type="source" position={Position.Right} className="!h-2.5 !w-2.5 !bg-neutral-400" />
      )}
    </div>
  );
}
