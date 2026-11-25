"use client";

import { useMemo } from "react";
import {
  Background,
  Controls,
  MarkerType,
  ReactFlow,
  ReactFlowProvider,
  type Edge,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { FlowNode } from "@/components/build/FlowNode";
import { isExpanded, neuronCount, type FlowNodeData } from "@/components/build/nodeCatalog";
import { useGraph } from "@/lib/store";

const nodeTypes: NodeTypes = { tcNode: FlowNode };

function ports(data: FlowNodeData, threshold: number, side: "s" | "t"): string[] {
  if (data.type === "linear" && isExpanded(data, threshold)) {
    const n = neuronCount(data) ?? 0;
    return Array.from({ length: n }, (_, i) => `${side}${i}`);
  }
  return [side === "s" ? "sc" : "tc"];
}

function Inner() {
  const nodes = useGraph((s) => s.nodes);
  const edges = useGraph((s) => s.edges);
  const threshold = useGraph((s) => s.renderThreshold);

  const renderEdges = useMemo<Edge[]>(() => {
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const out: Edge[] = [];
    for (const le of edges) {
      const a = byId[le.source];
      const b = byId[le.target];
      if (!a || !b) continue;
      for (const sh of ports(a.data, threshold, "s")) {
        for (const th of ports(b.data, threshold, "t")) {
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
  }, [nodes, edges, threshold]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={renderEdges}
      nodeTypes={nodeTypes}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      fitView
      proOptions={{ hideAttribution: true }}
    >
      <Background />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}

export function NetworkCanvas() {
  return (
    <ReactFlowProvider>
      <Inner />
    </ReactFlowProvider>
  );
}
