"use client";

import type { Node } from "@xyflow/react";
import {
  ACTIVATIONS,
  CATALOG_BY_TYPE,
  type Activation,
  type FlowNodeData,
} from "./nodeCatalog";
import { WeightMatrix } from "./WeightMatrix";

export interface NodeInspectorProps {
  node: Node<FlowNodeData> | null;
  onChange: (id: string, patch: Partial<FlowNodeData>) => void;
  onDelete: (id: string) => void;
}

function Labeled({
  label,
  help,
  children,
}: {
  label: string;
  help?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-neutral-300">{label}</span>
      {children}
      {help && <span className="mt-1 block text-[11px] text-neutral-500">{help}</span>}
    </label>
  );
}

const inputCls =
  "w-full rounded border border-neutral-700 bg-neutral-800 px-2 py-1.5 text-sm text-neutral-100 outline-none focus:border-sky-500";

export function NodeInspector({ node, onChange, onDelete }: NodeInspectorProps) {
  if (!node) {
    return (
      <div className="p-4 text-sm text-neutral-500">
        <p className="font-medium text-neutral-400">No node selected</p>
        <p className="mt-2 leading-relaxed">
          Right-click the canvas to add a node. Click a node to edit its attributes.
          Drag from a node&apos;s right handle to another&apos;s left handle to connect.
        </p>
      </div>
    );
  }

  const data = node.data;
  const entry = CATALOG_BY_TYPE[data.type];
  const setParam = (key: string, value: string) =>
    onChange(node.id, { params: { ...data.params, [key]: value } });

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-neutral-800 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-neutral-100">{data.label}</h3>
          <span className="font-mono text-[11px] text-neutral-500">{node.id}</span>
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {(entry?.fields ?? []).map((f) => (
          <Labeled key={f.key} label={f.label} help={f.help}>
            <input
              className={inputCls}
              inputMode="numeric"
              value={data.params[f.key] ?? ""}
              placeholder={f.placeholder}
              onChange={(e) => setParam(f.key, e.target.value)}
            />
          </Labeled>
        ))}

        {entry?.activation && (
          <Labeled label="Activation">
            <select
              className={inputCls}
              value={data.activation ?? "none"}
              onChange={(e) => onChange(node.id, { activation: e.target.value as Activation })}
            >
              {ACTIVATIONS.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </Labeled>
        )}

        {entry?.bias && (
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={data.bias ?? true}
              onChange={(e) => onChange(node.id, { bias: e.target.checked })}
              className="h-4 w-4 accent-sky-500"
            />
            <span className="text-xs font-medium text-neutral-300">Bias</span>
          </label>
        )}

        {data.type === "linear" && (
          <div>
            <span className="mb-1 block text-xs font-medium text-neutral-300">Weights</span>
            <WeightMatrix
              data={data}
              onSetInit={(initWeights, initBias) => onChange(node.id, { initWeights, initBias })}
            />
          </div>
        )}

        {data.error && (
          <div className="rounded border border-red-500/50 bg-red-500/10 p-2 text-[12px] leading-snug text-red-300">
            {data.error}
          </div>
        )}
        {data.outputShape && !data.error && (
          <div className="rounded border border-emerald-500/40 bg-emerald-500/10 p-2 font-mono text-[12px] text-emerald-300">
            output shape → [{data.outputShape.join(", ")}]
          </div>
        )}
      </div>

      <div className="border-t border-neutral-800 p-4">
        <button
          onClick={() => onDelete(node.id)}
          className="w-full rounded border border-red-500/40 px-3 py-1.5 text-sm text-red-300 hover:bg-red-500/10"
        >
          Delete node
        </button>
      </div>
    </div>
  );
}
