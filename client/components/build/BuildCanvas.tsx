"use client";

import { useCallback, useMemo, useState } from "react";
import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
  type EdgeChange,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { buildModel, type BuildResult, type BuildSuccess } from "@/lib/api";
import { canvasToSpec } from "@/lib/canvasToSpec";
import {
  defaultDataFor,
  isExpanded,
  neuronCount,
  type FlowNodeData,
  type PaletteItem,
} from "./nodeCatalog";
import { FlowNode } from "./FlowNode";
import { NodeInspector } from "./NodeInspector";
import { PaletteMenu } from "./PaletteMenu";
import { useGraph, type LogicalEdge, type TCNode } from "@/lib/store";

const nodeTypes: NodeTypes = { tcNode: FlowNode };

interface MenuState {
  screenX: number;
  screenY: number;
  flowX: number;
  flowY: number;
}

type CompileState =
  | null
  | { kind: "success"; data: BuildSuccess }
  | { kind: "error"; errors: { message: string; node_id: string | null }[] };

function sourcePorts(data: FlowNodeData, threshold: number): string[] {
  if (data.type === "linear" && isExpanded(data, threshold)) {
    const n = neuronCount(data) ?? 0;
    return Array.from({ length: n }, (_, i) => `s${i}`);
  }
  return ["sc"];
}
function targetPorts(data: FlowNodeData, threshold: number): string[] {
  if (data.type === "linear" && isExpanded(data, threshold)) {
    const n = neuronCount(data) ?? 0;
    return Array.from({ length: n }, (_, i) => `t${i}`);
  }
  return ["tc"];
}

function BuildCanvasInner() {
  const {
    nodes,
    edges,
    renderThreshold,
    modelName,
    onNodesChange,
    addNode,
    updateNodeData,
    deleteNode,
    selectNode,
    connect,
    removeEdges,
    setModelName,
    replaceGraph,
    applyResult,
  } = useGraph();

  const [menu, setMenu] = useState<MenuState | null>(null);
  const [result, setResult] = useState<CompileState>(null);
  const [compiling, setCompiling] = useState(false);
  const { screenToFlowPosition } = useReactFlow();

  const selectedNode = useMemo(() => nodes.find((n) => n.selected) ?? null, [nodes]);

  // Derive the rendered (possibly bipartite) edges from logical layer edges.
  const renderEdges = useMemo<Edge[]>(() => {
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const out: Edge[] = [];
    for (const le of edges) {
      const a = byId[le.source];
      const b = byId[le.target];
      if (!a || !b) continue;
      for (const sh of sourcePorts(a.data, renderThreshold)) {
        for (const th of targetPorts(b.data, renderThreshold)) {
          out.push({
            id: `${le.id}::${sh}>${th}`,
            source: le.source,
            target: le.target,
            sourceHandle: sh,
            targetHandle: th,
            type: "default",
            markerEnd: { type: MarkerType.ArrowClosed },
          });
        }
      }
    }
    return out;
  }, [nodes, edges, renderThreshold]);

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
      if (removed.length) {
        const logicalIds = [...new Set(removed.map((id) => id.split("::")[0]))];
        removeEdges(logicalIds);
      }
    },
    [removeEdges],
  );

  const onPaneContextMenu = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      e.preventDefault();
      const flow = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      setMenu({ screenX: e.clientX, screenY: e.clientY, flowX: flow.x, flowY: flow.y });
    },
    [screenToFlowPosition],
  );

  const onPick = useCallback(
    (item: PaletteItem) => {
      if (!menu) return;
      addNode(item.type, { x: menu.flowX, y: menu.flowY }, item.preset);
      setMenu(null);
    },
    [menu, addNode],
  );

  const clearAll = useCallback(() => {
    replaceGraph([], []);
    setResult(null);
  }, [replaceGraph]);

  const compile = useCallback(async () => {
    setCompiling(true);
    const res: BuildResult = await buildModel(canvasToSpec(nodes, edges, modelName));
    applyResult(res);
    if (res.ok) setResult({ kind: "success", data: res });
    else setResult({ kind: "error", errors: res.errors.map((e) => ({ message: e.message, node_id: e.node_id })) });
    setCompiling(false);
  }, [nodes, edges, modelName, applyResult]);

  const loadExample = useCallback(() => {
    const mk = (id: string, type: string, x: number, y: number, patch?: Partial<FlowNodeData>): TCNode => ({
      id,
      type: "tcNode",
      position: { x, y },
      data: defaultDataFor(type, patch),
      selected: false,
    });
    const ns = [
      mk("input-1", "input", 0, 160),
      mk("flatten-2", "flatten", 200, 180),
      mk("linear-3", "linear", 380, 160, { params: { out_features: "128" }, activation: "relu" }),
      mk("linear-4", "linear", 600, 160, { params: { out_features: "3" }, activation: "none" }),
      mk("output-5", "output", 820, 180, { activation: "softmax" }),
    ];
    const es: LogicalEdge[] = ns.slice(0, -1).map((n, i) => ({
      id: `le-${n.id}-${ns[i + 1].id}`,
      source: n.id,
      target: ns[i + 1].id,
    }));
    replaceGraph(ns, es);
    setResult(null);
  }, [replaceGraph]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-3 border-b border-neutral-800 bg-neutral-950 px-4 py-2">
        <input
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          className="w-40 rounded border border-neutral-700 bg-neutral-800 px-2 py-1 text-sm text-neutral-100 outline-none focus:border-sky-500"
          aria-label="model name"
        />
        <button
          onClick={compile}
          disabled={compiling || nodes.length === 0}
          className="rounded bg-sky-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-40"
        >
          {compiling ? "Compiling…" : "Compile"}
        </button>
        <button
          onClick={loadExample}
          className="rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800"
        >
          Load example
        </button>
        <button
          onClick={clearAll}
          className="rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800"
        >
          Clear
        </button>
        <span className="ml-auto text-xs text-neutral-500">Right-click the canvas to add a node</span>
      </div>

      <div className="flex min-h-0 flex-1">
        <div className="relative min-h-0 flex-1">
          <ReactFlow
            nodes={nodes}
            edges={renderEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={connect}
            onPaneContextMenu={onPaneContextMenu}
            nodeTypes={nodeTypes}
            deleteKeyCode={["Backspace", "Delete"]}
            fitView
            proOptions={{ hideAttribution: true }}
          >
            <Background />
            <Controls />
            <MiniMap pannable zoomable className="!bg-neutral-800" />
          </ReactFlow>

          {menu && (
            <PaletteMenu x={menu.screenX} y={menu.screenY} onPick={onPick} onClose={() => setMenu(null)} />
          )}

          {result && (
            <div className="absolute bottom-3 left-3 max-w-md rounded-lg border border-neutral-700 bg-neutral-900/95 p-3 text-sm shadow-xl">
              {result.kind === "success" ? (
                <div>
                  <div className="font-semibold text-emerald-400">✓ Network compiled</div>
                  <div className="mt-1 text-neutral-300">
                    <span className="font-mono text-neutral-400">{result.data.model_id}</span>
                    {" · "}
                    {result.data.num_parameters.toLocaleString()} parameters
                  </div>
                </div>
              ) : (
                <div>
                  <div className="font-semibold text-red-400">
                    ✗ {result.errors.length} problem{result.errors.length > 1 ? "s" : ""}
                  </div>
                  <ul className="mt-1 space-y-1">
                    {result.errors.map((e, i) => (
                      <li key={i}>
                        <button
                          disabled={!e.node_id}
                          onClick={() => e.node_id && selectNode(e.node_id)}
                          className="text-left text-red-300 enabled:hover:underline"
                        >
                          {e.node_id && <span className="font-mono text-[11px] text-red-400">[{e.node_id}] </span>}
                          {e.message}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        <aside className="w-80 shrink-0 border-l border-neutral-800 bg-neutral-950">
          <NodeInspector node={selectedNode} onChange={updateNodeData} onDelete={deleteNode} />
        </aside>
      </div>
    </div>
  );
}

// local helper so loadExample stays pure of hook calls
export function BuildCanvas() {
  return (
    <ReactFlowProvider>
      <BuildCanvasInner />
    </ReactFlowProvider>
  );
}
