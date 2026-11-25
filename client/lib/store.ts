"use client";

import { create } from "zustand";
import { type Connection, type Node, type NodeChange } from "@xyflow/react";
import { defaultDataFor, type FlowNodeData } from "@/components/build/nodeCatalog";
import type { BuildResult } from "@/lib/api";

export type TCNode = Node<FlowNodeData, "tcNode">;
export interface LogicalEdge {
  id: string;
  source: string;
  target: string;
}

interface Snapshot {
  nodes: TCNode[];
  edges: LogicalEdge[];
}

const STORAGE_KEY = "tensorcanvas:graph";
const HISTORY_LIMIT = 100;

// Largest numeric suffix among node ids (e.g. "dense-5" → 5), 0 if none. Used to
// keep generated ids from colliding with loaded graphs (examples use fixed ids
// like "input-1", so a freshly-added node must start past them).
function maxSuffix(nodes: { id: string }[]): number {
  let m = 0;
  for (const n of nodes) {
    const r = /-(\d+)$/.exec(n.id);
    if (r) m = Math.max(m, Number(r[1]));
  }
  return m;
}

// Push the current graph onto the undo stack and clear redo.
function pushHistory(s: { past: Snapshot[]; nodes: TCNode[]; edges: LogicalEdge[] }) {
  return {
    past: [...s.past, { nodes: s.nodes, edges: s.edges }].slice(-HISTORY_LIMIT),
    future: [] as Snapshot[],
  };
}

function applyNodeChangesLocal(changes: NodeChange<TCNode>[], nodes: TCNode[]): TCNode[] {
  if (changes.length === 0) return nodes;
  const map = new Map(nodes.map((n) => [n.id, n] as const));
  const added: TCNode[] = [];
  for (const c of changes) {
    switch (c.type) {
      case "position": {
        const n = map.get(c.id);
        if (n) map.set(c.id, { ...n, position: c.position ?? n.position, dragging: c.dragging });
        break;
      }
      case "dimensions": {
        const n = map.get(c.id);
        if (n && c.dimensions) map.set(c.id, { ...n, measured: c.dimensions });
        break;
      }
      case "select": {
        const n = map.get(c.id);
        if (n) map.set(c.id, { ...n, selected: c.selected });
        break;
      }
      case "remove":
        map.delete(c.id);
        break;
      case "add":
        added.push(c.item);
        break;
      case "replace":
        map.set(c.id, c.item);
        break;
    }
  }
  const result: TCNode[] = [];
  for (const n of nodes) {
    const u = map.get(n.id);
    if (u) result.push(u);
  }
  for (const a of added) if (!nodes.some((o) => o.id === a.id)) result.push(a);
  return result;
}

interface GraphState {
  nodes: TCNode[];
  edges: LogicalEdge[];
  renderThreshold: number;
  modelName: string;
  lastModelId: string | null;
  nextId: number;
  past: Snapshot[];
  future: Snapshot[];
  loaded: boolean;

  onNodesChange: (changes: NodeChange<TCNode>[]) => void;
  addNode: (type: string, pos: { x: number; y: number }, preset?: Partial<FlowNodeData>) => void;
  updateNodeData: (id: string, patch: Partial<FlowNodeData>) => void;
  deleteNode: (id: string) => void;
  deleteSelection: (nodeIds: string[], edgeIds: string[]) => void;
  selectNode: (id: string) => void;
  connect: (c: Connection) => void;
  removeEdges: (ids: string[]) => void;
  setThreshold: (n: number) => void;
  setModelName: (s: string) => void;
  replaceGraph: (nodes: TCNode[], edges: LogicalEdge[]) => void;
  applyResult: (res: BuildResult) => void;
  recordNow: () => void;
  undo: () => void;
  redo: () => void;
  save: () => void;
  hydrate: () => void;
}

export const useGraph = create<GraphState>((set, get) => ({
  nodes: [],
  edges: [],
  renderThreshold: 5,
  modelName: "untitled",
  lastModelId: null,
  nextId: 1,
  past: [],
  future: [],
  loaded: false,

  onNodesChange: (changes) =>
    set((s) => {
      const nodes = applyNodeChangesLocal(changes, s.nodes);
      const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
      const edges = removed.length
        ? s.edges.filter((e) => !removed.includes(e.source) && !removed.includes(e.target))
        : s.edges;
      return { nodes, edges };
    }),

  addNode: (type, pos, preset) =>
    set((s) => {
      // Collision-proof id: start past both nextId and any suffix already in the
      // graph (loaded examples/imports use fixed ids), then skip any still in use.
      const used = new Set(s.nodes.map((n) => n.id));
      let k = Math.max(s.nextId, maxSuffix(s.nodes) + 1);
      while (used.has(`${type}-${k}`)) k += 1;
      const id = `${type}-${k}`;
      const node: TCNode = { id, type: "tcNode", position: pos, data: defaultDataFor(type, preset), selected: true };
      return {
        ...pushHistory(s),
        nodes: [...s.nodes.map((n) => ({ ...n, selected: false })), node],
        nextId: k + 1,
      };
    }),

  updateNodeData: (id, patch) =>
    set((s) => {
      // Editing a node's structure makes the last compile's artifacts stale.
      // Clear them so the canvas doesn't show weights/shapes that no longer match
      // (and would otherwise plant a guaranteed failure on the next compile).
      const STRUCTURAL = ["type", "params", "activation", "bias", "customType", "extra", "neuronActivations"];
      const invalidate = Object.keys(patch).some((k) => STRUCTURAL.includes(k))
        ? { outputShape: null, weights: null, weightShape: null, hasBias: null, biasValues: null, error: null }
        : {};
      return { nodes: s.nodes.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...invalidate, ...patch } } : n)) };
    }),

  deleteNode: (id) =>
    set((s) => ({
      ...pushHistory(s),
      nodes: s.nodes.filter((n) => n.id !== id),
      edges: s.edges.filter((e) => e.source !== id && e.target !== id),
    })),

  deleteSelection: (nodeIds, edgeIds) =>
    set((s) => {
      if (nodeIds.length === 0 && edgeIds.length === 0) return {};
      const dead = new Set(nodeIds);
      return {
        ...pushHistory(s),
        nodes: s.nodes.filter((n) => !dead.has(n.id)),
        edges: s.edges.filter(
          (e) => !edgeIds.includes(e.id) && !dead.has(e.source) && !dead.has(e.target),
        ),
      };
    }),

  selectNode: (id) => set((s) => ({ nodes: s.nodes.map((n) => ({ ...n, selected: n.id === id })) })),

  connect: (c) =>
    set((s) => {
      if (!c.source || !c.target || c.source === c.target) return {};
      if (s.edges.some((e) => e.source === c.source && e.target === c.target)) return {};
      return {
        ...pushHistory(s),
        edges: [...s.edges, { id: `le-${c.source}-${c.target}`, source: c.source, target: c.target }],
      };
    }),

  removeEdges: (ids) =>
    set((s) => (ids.length ? { ...pushHistory(s), edges: s.edges.filter((e) => !ids.includes(e.id)) } : {})),

  setThreshold: (n) => set({ renderThreshold: n }),
  setModelName: (s) => set({ modelName: s }),

  replaceGraph: (nodes, edges) =>
    set((s) => ({ ...pushHistory(s), nodes, edges, nextId: Math.max(s.nextId, maxSuffix(nodes) + 1) })),

  recordNow: () => set((s) => pushHistory(s)),

  undo: () =>
    set((s) => {
      if (s.past.length === 0) return {};
      const prev = s.past[s.past.length - 1];
      return {
        nodes: prev.nodes,
        edges: prev.edges,
        past: s.past.slice(0, -1),
        future: [{ nodes: s.nodes, edges: s.edges }, ...s.future].slice(0, HISTORY_LIMIT),
      };
    }),

  redo: () =>
    set((s) => {
      if (s.future.length === 0) return {};
      const next = s.future[0];
      return {
        nodes: next.nodes,
        edges: next.edges,
        future: s.future.slice(1),
        past: [...s.past, { nodes: s.nodes, edges: s.edges }].slice(-HISTORY_LIMIT),
      };
    }),

  save: () => {
    if (typeof window === "undefined") return;
    const s = get();
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        nodes: s.nodes,
        edges: s.edges,
        modelName: s.modelName,
        renderThreshold: s.renderThreshold,
        nextId: s.nextId,
      }),
    );
  },

  hydrate: () => {
    if (get().loaded || typeof window === "undefined") return;
    set({ loaded: true });
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {
      const d = JSON.parse(raw);
      set({
        nodes: d.nodes ?? [],
        edges: d.edges ?? [],
        modelName: d.modelName ?? "untitled",
        renderThreshold: d.renderThreshold ?? 5,
        nextId: Math.max(d.nextId ?? 1, maxSuffix(d.nodes ?? []) + 1),
      });
    } catch {
      /* ignore corrupt storage */
    }
  },

  applyResult: (res) =>
    set((s) => {
      if (res.ok) {
        const byId = Object.fromEntries(res.nodes.map((n) => [n.id, n]));
        return {
          lastModelId: res.model_id,
          nodes: s.nodes.map((n) => {
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
        };
      }
      const byNode: Record<string, string[]> = {};
      for (const e of res.errors) if (e.node_id) (byNode[e.node_id] ??= []).push(e.message);
      return {
        nodes: s.nodes.map((n) => ({
          ...n,
          data: { ...n.data, outputShape: null, error: byNode[n.id]?.join(" ") ?? null },
        })),
      };
    }),
}));
