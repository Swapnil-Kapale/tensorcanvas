"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import {
  addEdge,
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
  type Node,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { buildModel, type BuildResult, type BuildSuccess } from "@/lib/api";
import { canvasToSpec } from "@/lib/canvasToSpec";
import { defaultDataFor, type FlowNodeData } from "./nodeCatalog";
import { FlowNode } from "./FlowNode";
import { NodeInspector } from "./NodeInspector";
import { PaletteMenu } from "./PaletteMenu";

type TCNode = Node<FlowNodeData, "tcNode">;

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

function BuildCanvasInner() {
  const [nodes, setNodes, onNodesChange] = useNodesState<TCNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [menu, setMenu] = useState<MenuState | null>(null);
  const [result, setResult] = useState<CompileState>(null);
  const [compiling, setCompiling] = useState(false);
  const [name, setName] = useState("untitled");
  const idRef = useRef(1);
  const { screenToFlowPosition } = useReactFlow();

  const selectedNode = useMemo(() => nodes.find((n) => n.selected) ?? null, [nodes]);

  const onConnect = useCallback(
    (c: Connection) => setEdges((eds) => addEdge(c, eds)),
    [setEdges],
  );

  const onPaneContextMenu = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      e.preventDefault();
      const flow = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      setMenu({ screenX: e.clientX, screenY: e.clientY, flowX: flow.x, flowY: flow.y });
    },
    [screenToFlowPosition],
  );

  const addNode = useCallback(
    (type: string) => {
      if (!menu) return;
      const id = `${type}-${idRef.current++}`;
      const newNode: TCNode = {
        id,
        type: "tcNode",
        position: { x: menu.flowX, y: menu.flowY },
        data: defaultDataFor(type),
        selected: true,
      };
      setNodes((nds) => [...nds.map((n) => ({ ...n, selected: false })), newNode]);
      setMenu(null);
    },
    [menu, setNodes],
  );

  const onChangeNode = useCallback(
    (id: string, patch: Partial<FlowNodeData>) =>
      setNodes((nds) =>
        nds.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...patch } } : n)),
      ),
    [setNodes],
  );

  const onDeleteNode = useCallback(
    (id: string) => {
      setNodes((nds) => nds.filter((n) => n.id !== id));
      setEdges((eds) => eds.filter((e) => e.source !== id && e.target !== id));
    },
    [setNodes, setEdges],
  );

  const selectNode = useCallback(
    (id: string) => setNodes((nds) => nds.map((n) => ({ ...n, selected: n.id === id }))),
    [setNodes],
  );

  const clearAll = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setResult(null);
  }, [setNodes, setEdges]);

  const applyResult = useCallback(
    (res: BuildResult) => {
      if (res.ok) {
        const byId = Object.fromEntries(res.nodes.map((n) => [n.id, n]));
        setNodes((nds) =>
          nds.map((n) => {
            const info = byId[n.id];
            return {
              ...n,
              data: {
                ...n.data,
                error: null,
                outputShape: info?.output_shape ?? null,
                weights: info?.weights ?? null,
                biasValues: info?.bias_values ?? null,
                weightShape: info?.weight_shape ?? null,
                hasBias: info?.has_bias ?? null,
              },
            };
          }),
        );
        setResult({ kind: "success", data: res });
      } else {
        const byNode: Record<string, string[]> = {};
        for (const e of res.errors) {
          if (e.node_id) (byNode[e.node_id] ??= []).push(e.message);
        }
        setNodes((nds) =>
          nds.map((n) => ({
            ...n,
            data: { ...n.data, outputShape: null, error: byNode[n.id]?.join(" ") ?? null },
          })),
        );
        setResult({
          kind: "error",
          errors: res.errors.map((e) => ({ message: e.message, node_id: e.node_id })),
        });
      }
    },
    [setNodes],
  );

  const compile = useCallback(async () => {
    setCompiling(true);
    const spec = canvasToSpec(nodes, edges, name);
    const res = await buildModel(spec);
    applyResult(res);
    setCompiling(false);
  }, [nodes, edges, name, applyResult]);

  const loadExample = useCallback(() => {
    idRef.current = 100;
    const mk = (type: string, x: number, y: number, patch?: Partial<FlowNodeData>): TCNode => ({
      id: `${type}-${idRef.current++}`,
      type: "tcNode",
      position: { x, y },
      data: { ...defaultDataFor(type), ...patch },
      selected: false,
    });
    const ns = [
      mk("input", 0, 140),
      mk("flatten", 190, 160),
      mk("linear", 360, 140, { params: { out_features: "128" }, activation: "relu" }),
      mk("linear", 560, 140, { params: { out_features: "10" }, activation: "none" }),
      mk("output", 760, 160, { activation: "softmax" }),
    ];
    const es: Edge[] = ns.slice(0, -1).map((n, i) => ({
      id: `e-${n.id}-${ns[i + 1].id}`,
      source: n.id,
      target: ns[i + 1].id,
      type: "default",
      markerEnd: { type: MarkerType.ArrowClosed },
    }));
    setNodes(ns);
    setEdges(es);
    setResult(null);
  }, [setNodes, setEdges]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-3 border-b border-neutral-800 bg-neutral-950 px-4 py-2">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
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
        <span className="ml-auto text-xs text-neutral-500">
          Right-click the canvas to add a node
        </span>
      </div>

      {/* Canvas + inspector */}
      <div className="flex min-h-0 flex-1">
        <div className="relative min-h-0 flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onPaneContextMenu={onPaneContextMenu}
            nodeTypes={nodeTypes}
            deleteKeyCode={["Backspace", "Delete"]}
            defaultEdgeOptions={{ type: "default", markerEnd: { type: MarkerType.ArrowClosed } }}
            fitView
            proOptions={{ hideAttribution: true }}
          >
            <Background />
            <Controls />
            <MiniMap pannable zoomable className="!bg-neutral-800" />
          </ReactFlow>

          {menu && (
            <PaletteMenu
              x={menu.screenX}
              y={menu.screenY}
              onPick={addNode}
              onClose={() => setMenu(null)}
            />
          )}

          {/* Results overlay */}
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
                          {e.node_id && (
                            <span className="font-mono text-[11px] text-red-400">
                              [{e.node_id}]{" "}
                            </span>
                          )}
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
          <NodeInspector node={selectedNode} onChange={onChangeNode} onDelete={onDeleteNode} />
        </aside>
      </div>
    </div>
  );
}

export function BuildCanvas() {
  return (
    <ReactFlowProvider>
      <BuildCanvasInner />
    </ReactFlowProvider>
  );
}
