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
  const exRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const addOffset = useRef(0);

  // Load the saved graph once (store guards against clobbering on route return).
  useEffect(() => {
    hydrate();
  }, [hydrate]);

  // Close the "Load example" menu on an outside click or Escape — NOT on
  // mouse-leave, which used to close it before a click on an item could land.
  useEffect(() => {
    if (!exMenu) return;
    const onDown = (e: MouseEvent) => {
      if (exRef.current && !exRef.current.contains(e.target as Node)) setExMenu(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setExMenu(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [exMenu]);

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
    "flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-sm text-[var(--muted)] transition hover:bg-[var(--glass-hover)] hover:text-[var(--text)] disabled:opacity-30 disabled:hover:bg-transparent";

  return (
    <div className="relative h-full min-h-0">
      {/* ── base canvas: borderless, immersive lowest layer ── */}
      <div
        ref={paneRef}
        className="tc-stage absolute inset-0"
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
          <Background color="rgba(255,255,255,0.07)" gap={26} />
          <Controls />
          <MiniMap pannable zoomable />
        </ReactFlow>
      </div>

      {/* ── floating sub-toolbar ── */}
      <div className="tc-glass absolute left-4 top-4 z-20 flex items-center gap-2 rounded-2xl px-3 py-2">
        <input
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          className="tc-field w-36 px-2 py-1 text-sm"
          aria-label="model name"
        />
        <button
          onClick={compile}
          disabled={compiling || nodes.length === 0}
          className="tc-btn-primary rounded-lg px-3 py-1.5 text-sm"
        >
          {compiling ? "Compiling…" : "Compile"}
        </button>
        <div className="relative" ref={exRef}>
          <button onClick={() => setExMenu((v) => !v)} className="tc-btn px-3 py-1.5 text-sm">
            Load example ▾
          </button>
          {exMenu && (
            <div className="tc-glass absolute left-0 top-full z-30 mt-2 w-56 overflow-hidden rounded-xl py-1">
              {EXAMPLES.map((grp) => (
                <div key={grp.group} className="py-0.5">
                  <div className="px-3 pt-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--faint)]">{grp.group}</div>
                  {grp.items.map((it) => (
                    <button
                      key={it.id}
                      onClick={() => loadGraph(it.make(), `Loaded ${it.label}`)}
                      className="block w-full px-3 py-1.5 text-left text-sm text-[var(--text)] hover:bg-[var(--glass-hover)]"
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
        <button onClick={() => fileRef.current?.click()} className="tc-btn px-3 py-1.5 text-sm">Import</button>
        <button onClick={exportGraph} disabled={nodes.length === 0} className="tc-btn px-3 py-1.5 text-sm">Export</button>
        <button onClick={clearAll} className="tc-btn px-3 py-1.5 text-sm">Clear</button>
      </div>

      <span className="pointer-events-none absolute right-[22rem] top-6 z-20 mr-4 hidden text-xs text-[var(--faint)] xl:block">
        Right-click to add · drag-drop a .tcnet.json
      </span>

      {/* ── floating bottom toolbar ── */}
      <div className="tc-glass absolute bottom-4 left-1/2 z-20 flex -translate-x-1/2 items-center gap-1 rounded-2xl p-1.5">
        <button
          className={tbBtn + " font-medium text-[var(--text)]"}
          onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            setAddMenu({ x: r.left, y: Math.max(8, r.top - 320) });
          }}
          title="Add a node"
        >
          <span className="text-[var(--accent)]">＋</span> Add
        </button>
        <div className="mx-1 h-5 w-px bg-[var(--stroke)]" />
        <button className={tbBtn} onClick={doDelete} title="Delete selection (⌫)">Delete</button>
        <button className={tbBtn} onClick={undo} disabled={past.length === 0} title="Undo (⌘Z)">↶</button>
        <button className={tbBtn} onClick={redo} disabled={future.length === 0} title="Redo (⌘⇧Z)">↷</button>
        <div className="mx-1 h-5 w-px bg-[var(--stroke)]" />
        <button className={tbBtn} onClick={saveNow} title="Save (⌘S)">
          {justSaved ? <span className="text-[var(--emerald)]">Saved ✓</span> : "Save"}
        </button>
      </div>

      {notice && (
        <div className="tc-glass absolute left-1/2 top-20 z-30 -translate-x-1/2 rounded-xl px-3 py-1.5 text-xs text-[var(--text)]">
          {notice}
        </div>
      )}

      {menu && <PaletteMenu x={menu.x} y={menu.y} onPick={onPickAt} onClose={() => setMenu(null)} />}
      {addMenu && <PaletteMenu x={addMenu.x} y={addMenu.y} onPick={addAtCenter} onClose={() => setAddMenu(null)} />}

      {result && (
        <div className="tc-glass absolute bottom-4 left-4 z-20 max-w-md rounded-xl p-3 text-sm">
          {result.kind === "success" ? (
            <div>
              <div className="font-semibold text-[var(--emerald)]">✓ Network compiled</div>
              <div className="mt-1 text-[var(--muted)]">
                <span className="font-mono text-[var(--faint)]">{result.data.model_id}</span>
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

      {/* ── floating inspector ── */}
      <aside className="tc-panel absolute bottom-4 right-4 top-4 z-20 w-80 overflow-auto">
        <NodeInspector node={selectedNode} onChange={updateNodeData} onDelete={deleteNode} />
      </aside>
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
