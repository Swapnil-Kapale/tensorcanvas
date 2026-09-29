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

// Minimal node-change applier — this @xyflow/react build does not export
// applyNodeChanges at runtime, so we handle the change types we rely on.
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

  onNodesChange: (changes: NodeChange<TCNode>[]) => void;
  addNode: (type: string, pos: { x: number; y: number }, preset?: Partial<FlowNodeData>) => void;
  updateNodeData: (id: string, patch: Partial<FlowNodeData>) => void;
  deleteNode: (id: string) => void;
  selectNode: (id: string) => void;
  connect: (c: Connection) => void;
  removeEdges: (ids: string[]) => void;
  setThreshold: (n: number) => void;
  setModelName: (s: string) => void;
  replaceGraph: (nodes: TCNode[], edges: LogicalEdge[]) => void;
  applyResult: (res: BuildResult) => void;
}

export const useGraph = create<GraphState>((set) => ({
  nodes: [],
  edges: [],
  renderThreshold: 5,
  modelName: "untitled",
  lastModelId: null,
  nextId: 1,

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
      const id = `${type}-${s.nextId}`;
      const node: TCNode = {
        id,
        type: "tcNode",
        position: pos,
        data: defaultDataFor(type, preset),
        selected: true,
      };
      return {
        nodes: [...s.nodes.map((n) => ({ ...n, selected: false })), node],
        nextId: s.nextId + 1,
      };
    }),

  updateNodeData: (id, patch) =>
    set((s) => ({
      nodes: s.nodes.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...patch } } : n)),
    })),

  deleteNode: (id) =>
    set((s) => ({
      nodes: s.nodes.filter((n) => n.id !== id),
      edges: s.edges.filter((e) => e.source !== id && e.target !== id),
    })),

  selectNode: (id) =>
    set((s) => ({ nodes: s.nodes.map((n) => ({ ...n, selected: n.id === id })) })),

  connect: (c) =>
    set((s) => {
      if (!c.source || !c.target || c.source === c.target) return {};
      if (s.edges.some((e) => e.source === c.source && e.target === c.target)) return {};
      return {
        edges: [
          ...s.edges,
          { id: `le-${c.source}-${c.target}`, source: c.source, target: c.target },
        ],
      };
    }),

  removeEdges: (ids) =>
    set((s) => ({ edges: s.edges.filter((e) => !ids.includes(e.id)) })),

  setThreshold: (n) => set({ renderThreshold: n }),
  setModelName: (s) => set({ modelName: s }),

  replaceGraph: (nodes, edges) => set({ nodes, edges }),

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
