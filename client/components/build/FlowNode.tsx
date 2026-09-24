"use client";

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import {
  ACTIVATION_GLYPH,
  CATALOG_BY_TYPE,
  neuronCount,
  type FlowNodeData,
} from "./nodeCatalog";

type TCNode = Node<FlowNodeData, "tcNode">;

const EXPAND_MAX = 5; // >5 neurons → collapse to a compact box

// Per-neuron weight magnitude (mean |incoming weight|), normalised to [0,1].
function neuronMagnitudes(weights: number[][] | null | undefined): number[] | null {
  if (!weights || weights.length === 0) return null;
  const means = weights.map((row) =>
    row.length ? row.reduce((a, b) => a + Math.abs(b), 0) / row.length : 0,
  );
  const max = Math.max(...means, 1e-9);
  return means.map((m) => Math.min(1, m / max));
}

function Neuron({ mag, size }: { mag: number | null; size: number }) {
  return (
    <div
      className="relative shrink-0 rounded-full border border-sky-400/70"
      style={{ height: size, width: size }}
    >
      <div
        className="absolute inset-[2px] rounded-full bg-sky-400"
        style={{ opacity: mag == null ? 0 : 0.15 + mag * 0.8 }}
      />
    </div>
  );
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
    <div className="flex items-center justify-between gap-2">
      <span className="text-sm font-semibold">{data.label}</span>
      <div className="flex items-center gap-1">
        <Glyph data={data} />
        {data.bias && (
          <span
            title="bias enabled"
            className="h-2 w-2 rounded-full bg-amber-400"
            aria-label="bias"
          />
        )}
      </div>
    </div>
  );
}

function Shell({
  data,
  selected,
  children,
}: {
  data: FlowNodeData;
  selected: boolean;
  children: React.ReactNode;
}) {
  const hasError = Boolean(data.error);
  const showTarget = data.type !== "input";
  const showSource = data.type !== "output";
  return (
    <div
      className={[
        "rounded-lg border-2 bg-neutral-900/95 px-3 py-2 text-neutral-100 shadow-lg",
        "min-w-[140px] transition-colors",
        hasError ? "border-red-500" : "border-neutral-600",
        selected ? "ring-2 ring-white/60" : "",
      ].join(" ")}
    >
      {showTarget && (
        <Handle type="target" position={Position.Left} className="!h-2.5 !w-2.5 !bg-neutral-400" />
      )}
      {children}
      {data.outputShape && !hasError && (
        <div className="mt-1 inline-block rounded bg-emerald-500/15 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
          → [{data.outputShape.join(", ")}]
        </div>
      )}
      {hasError && (
        <div className="mt-1 max-w-[220px] text-[11px] leading-snug text-red-300">{data.error}</div>
      )}
      {showSource && (
        <Handle type="source" position={Position.Right} className="!h-2.5 !w-2.5 !bg-neutral-400" />
      )}
    </div>
  );
}

function DenseBody({ data }: { data: FlowNodeData }) {
  const n = neuronCount(data) ?? 0;
  const mags = neuronMagnitudes(data.weights);
  if (n > 0 && n <= EXPAND_MAX) {
    return (
      <div className="mt-2 flex flex-col items-center gap-1.5">
        {Array.from({ length: n }).map((_, i) => (
          <Neuron key={i} mag={mags?.[i] ?? null} size={26} />
        ))}
      </div>
    );
  }
  // Collapsed: 3 sample circles + count
  return (
    <div className="mt-2 flex flex-col items-center gap-1">
      <div className="flex gap-1.5">
        {[0, 1, 2].map((i) => (
          <Neuron key={i} mag={mags?.[i] ?? null} size={16} />
        ))}
      </div>
      <div className="text-center leading-tight">
        <div className="text-lg font-bold">{n || "—"}</div>
        <div className="text-[10px] text-neutral-400">neurons</div>
      </div>
    </div>
  );
}

function summary(data: FlowNodeData): string | null {
  if (data.type === "conv2d") {
    const { out_channels, kernel_size } = data.params;
    return `${out_channels ?? "—"} × ${kernel_size ?? "—"}×${kernel_size ?? "—"}`;
  }
  if (data.type === "maxpool2d") return `k=${data.params.kernel_size ?? "—"}`;
  if (data.type === "input") return `[${(data.params.shape ?? "").trim()}]`;
  return null;
}

export function FlowNode({ data, selected }: NodeProps<TCNode>) {
  const entry = CATALOG_BY_TYPE[data.type];
  const body =
    data.type === "linear" ? (
      <DenseBody data={data} />
    ) : (
      (() => {
        const s = summary(data);
        return s ? <div className="mt-1 font-mono text-[11px] text-neutral-400">{s}</div> : null;
      })()
    );

  return (
    <Shell data={data} selected={!!selected}>
      <Header data={data} />
      {body}
      <span className="sr-only">{entry?.type}</span>
    </Shell>
  );
}
