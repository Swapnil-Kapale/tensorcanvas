"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  isExpanded,
  neuronCount,
  type FlowNodeData,
  type PaletteItem,
} from "./nodeCatalog";
import { FlowNode } from "./FlowNode";
import { NodeInspector } from "./NodeInspector";
import { PaletteMenu } from "./PaletteMenu";
import { useGraph, type LogicalEdge, type TCNode } from "@/lib/store";
import { EXAMPLES, fileToGraph, graphToFile } from "@/lib/examples";

const nodeTypes: NodeTypes = { tcNode: FlowNode };

interface MenuState {
  x: number;
  y: number;
  flowX?: number;
  flowY?: number;
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
    past,
    future,
    onNodesChange,
    addNode,
    updateNodeData,
    deleteNode,
    deleteSelection,
    selectNode,
    connect,
    removeEdges,
    setModelName,
    replaceGraph,
    applyResult,
    recordNow,
    undo,
    redo,
    save,
    hydrate,
  } = useGraph();

  const [menu, setMenu] = useState<MenuState | null>(null);
  const [addMenu, setAddMenu] = useState<MenuState | null>(null);
  const [result, setResult] = useState<CompileState>(null);
  const [compiling, setCompiling] = useState(false);
  const [justSaved, setJustSaved] = useState(false);
  const [selEdge, setSelEdge] = useState<string | null>(null);
  const [exMenu, setExMenu] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const { screenToFlowPosition, getNodes } = useReactFlow();
  const paneRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const addOffset = useRef(0);

  // Load the saved graph once (store guards against clobbering on route return).
  useEffect(() => {
    hydrate();
  }, [hydrate]);

  const selectedNode = useMemo(() => nodes.find((n) => n.selected) ?? null, [nodes]);

  const renderEdges = useMemo<Edge[]>(() => {
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const out: Edge[] = [];
    for (const le of edges) {
      const a = byId[le.source];
      const b = byId[le.target];
      if (!a || !b) continue;
      for (const sh of sourcePorts(a.data, renderThreshold)) {
        for (const th of targetPorts(b.data, renderThreshold)) {
          const selected = le.id === selEdge;
          out.push({
            id: `${le.id}::${sh}>${th}`,
            source: le.source,
            target: le.target,
            sourceHandle: sh,
            targetHandle: th,
            type: "default",
            markerEnd: { type: MarkerType.ArrowClosed },
            selected,
            style: selected ? { stroke: "#38bdf8", strokeWidth: 2.5 } : undefined,
          });
        }
      }
    }
    return out;
  }, [nodes, edges, renderThreshold, selEdge]);

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
      if (removed.length) removeEdges([...new Set(removed.map((id) => id.split("::")[0]))]);
    },
    [removeEdges],
  );

  const onPaneContextMenu = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      e.preventDefault();
      const flow = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      setMenu({ x: e.clientX, y: e.clientY, flowX: flow.x, flowY: flow.y });
    },
    [screenToFlowPosition],
  );

  const onPickAt = useCallback(
    (item: PaletteItem) => {
      if (!menu) return;
      addNode(item.type, { x: menu.flowX ?? 0, y: menu.flowY ?? 0 }, item.preset);
      setMenu(null);
    },
    [menu, addNode],
  );

  const addAtCenter = useCallback(
    (item: PaletteItem) => {
      let pos = { x: 240, y: 160 };
      const el = paneRef.current;
      if (el) {
        const r = el.getBoundingClientRect();
        pos = screenToFlowPosition({ x: r.left + r.width / 2, y: r.top + r.height / 2 });
      }
      const k = addOffset.current++ % 6;
      addNode(item.type, { x: pos.x + k * 28, y: pos.y + k * 28 }, item.preset);
      setAddMenu(null);
    },
    [screenToFlowPosition, addNode],
  );

  const doDelete = useCallback(() => {
    const selNodes = getNodes().filter((n) => n.selected).map((n) => n.id);
    const edgeIds = selEdge ? [selEdge] : [];
    deleteSelection(selNodes, edgeIds);
    setSelEdge(null);
  }, [getNodes, selEdge, deleteSelection]);

  const saveNow = useCallback(() => {
    save();
    setJustSaved(true);
    window.setTimeout(() => setJustSaved(false), 1500);
  }, [save]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      const t = e.target as HTMLElement | null;
      const inField =
        !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
      if (mod && e.key.toLowerCase() === "s") {
        e.preventDefault();
        saveNow();
      } else if (mod && e.key.toLowerCase() === "z") {
        if (inField) return;
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if (mod && e.key.toLowerCase() === "y") {
        if (inField) return;
        e.preventDefault();
        redo();
      } else if (e.key === "Delete" || e.key === "Backspace") {
        if (inField) return;
        e.preventDefault();
        doDelete();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [saveNow, undo, redo, doDelete]);

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

  const loadGraph = useCallback(
    (g: { nodes: TCNode[]; edges: LogicalEdge[] }, label?: string) => {
      replaceGraph(g.nodes, g.edges);
      setResult(null);
      setExMenu(false);
      if (label) {
        setNotice(label);
        window.setTimeout(() => setNotice(null), 2500);
      }
    },
    [replaceGraph],
  );

  const exportGraph = useCallback(() => {
    const file = graphToFile(nodes, edges, modelName || "network");
    const blob = new Blob([JSON.stringify(file, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${(modelName || "network").replace(/\s+/g, "-")}.tcnet.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [nodes, edges, modelName]);

  const importFile = useCallback(
    async (file: File) => {
      try {
        const g = fileToGraph(JSON.parse(await file.text()));
        loadGraph(g, `Loaded ${file.name}`);
      } catch (e) {
        setNotice(e instanceof Error ? e.message : "Could not import file");
        window.setTimeout(() => setNotice(null), 3500);
      }
    },
    [loadGraph],
  );

  const tbBtn =
    "flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-sm text-neutral-300 transition hover:bg-neutral-800 disabled:opacity-30 disabled:hover:bg-transparent";

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
        <div className="relative">
          <button
            onClick={() => setExMenu((v) => !v)}
            className="rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800"
          >
            Load example ▾
          </button>
          {exMenu && (
            <div
              className="absolute left-0 top-full z-30 mt-1 w-56 overflow-hidden rounded-lg border border-neutral-700 bg-neutral-900 py-1 shadow-2xl"
              onMouseLeave={() => setExMenu(false)}
            >
              {EXAMPLES.map((grp) => (
                <div key={grp.group} className="py-0.5">
                  <div className="px-3 pt-1 text-[10px] font-semibold uppercase tracking-wide text-neutral-600">{grp.group}</div>
                  {grp.items.map((it) => (
                    <button
                      key={it.id}
                      onClick={() => loadGraph(it.make(), `Loaded ${it.label}`)}
                      className="block w-full px-3 py-1.5 text-left text-sm text-neutral-200 hover:bg-neutral-700/70"
                    >
                      {it.label}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".json,.tcnet,.tcnet.json"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && importFile(e.target.files[0])}
        />
        <button onClick={() => fileRef.current?.click()} className="rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800">
          Import
        </button>
        <button onClick={exportGraph} disabled={nodes.length === 0} className="rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800 disabled:opacity-40">
          Export
        </button>
        <button onClick={clearAll} className="rounded border border-neutral-700 px-3 py-1.5 text-sm text-neutral-300 hover:bg-neutral-800">
          Clear
        </button>
        <span className="ml-auto text-xs text-neutral-500">Right-click to add · drag-drop a .tcnet.json to load</span>
      </div>

      <div className="flex min-h-0 flex-1">
        <div
          ref={paneRef}
          className="relative min-h-0 flex-1"
          onDragOver={(e) => {
            if (e.dataTransfer.types.includes("Files")) e.preventDefault();
          }}
          onDrop={(e) => {
            if (e.dataTransfer.files?.[0]) {
              e.preventDefault();
              importFile(e.dataTransfer.files[0]);
            }
          }}
        >
          <ReactFlow
            nodes={nodes}
            edges={renderEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={connect}
            onNodeDragStart={() => recordNow()}
            onEdgeClick={(_, edge) => setSelEdge(edge.id.split("::")[0])}
            onNodeClick={() => setSelEdge(null)}
            onPaneClick={() => setSelEdge(null)}
            onPaneContextMenu={onPaneContextMenu}
            nodeTypes={nodeTypes}
            deleteKeyCode={null}
            fitView
            proOptions={{ hideAttribution: true }}
          >
            <Background />
            <Controls />
            <MiniMap pannable zoomable className="!bg-neutral-800" />
          </ReactFlow>

          {/* Floating canvas toolbar */}
          <div className="absolute bottom-4 left-1/2 z-20 flex -translate-x-1/2 items-center gap-1 rounded-xl border border-neutral-700/80 bg-neutral-900/90 p-1.5 shadow-2xl backdrop-blur">
            <button
              className={tbBtn + " font-medium text-neutral-100"}
              onClick={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                setAddMenu({ x: r.left, y: Math.max(8, r.top - 320) });
              }}
              title="Add a node"
            >
              <span className="text-sky-400">＋</span> Add
            </button>
            <div className="mx-1 h-5 w-px bg-neutral-700" />
            <button className={tbBtn} onClick={doDelete} title="Delete selection (⌫)">
              Delete
            </button>
            <button className={tbBtn} onClick={undo} disabled={past.length === 0} title="Undo (⌘Z)">
              ↶
            </button>
            <button className={tbBtn} onClick={redo} disabled={future.length === 0} title="Redo (⌘⇧Z)">
              ↷
            </button>
            <div className="mx-1 h-5 w-px bg-neutral-700" />
            <button className={tbBtn} onClick={saveNow} title="Save (⌘S)">
              {justSaved ? <span className="text-emerald-400">Saved ✓</span> : "Save"}
            </button>
          </div>

          {notice && (
            <div className="absolute left-1/2 top-3 z-30 -translate-x-1/2 rounded-lg border border-neutral-700 bg-neutral-900/95 px-3 py-1.5 text-xs text-neutral-200 shadow-xl">
              {notice}
            </div>
          )}

          {menu && (
            <PaletteMenu x={menu.x} y={menu.y} onPick={onPickAt} onClose={() => setMenu(null)} />
          )}
          {addMenu && (
            <PaletteMenu x={addMenu.x} y={addMenu.y} onPick={addAtCenter} onClose={() => setAddMenu(null)} />
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

export function BuildCanvas() {
  return (
    <ReactFlowProvider>
      <BuildCanvasInner />
    </ReactFlowProvider>
  );
}
