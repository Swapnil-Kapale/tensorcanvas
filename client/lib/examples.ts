// Example networks + the .tcnet.json file format (architecture import/export).
//
// A .tcnet.json file is just the canvas graph: nodes (type + params + position)
// and edges. It round-trips through the Build canvas and maps 1:1 to the build
// endpoint. (Weights are NOT included — architecture only.)

import { defaultDataFor, type FlowNodeData } from "@/components/build/nodeCatalog";
import type { LogicalEdge, TCNode } from "@/lib/store";

export const NET_FORMAT = "tensorcanvas-net";
export const NET_VERSION = 1;

export interface NetFileNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  params?: Record<string, string>;
  activation?: string;
  bias?: boolean;
  customType?: string;
}
export interface NetFile {
  format: string;
  version: number;
  name: string;
  nodes: NetFileNode[];
  edges: { source: string; target: string }[];
}

export interface Graph {
  nodes: TCNode[];
  edges: LogicalEdge[];
}

function mk(id: string, type: string, x: number, y: number, patch?: Partial<FlowNodeData>): TCNode {
  return { id, type: "tcNode", position: { x, y }, data: defaultDataFor(type, patch), selected: false };
}

function chain(ns: TCNode[]): LogicalEdge[] {
  return ns.slice(0, -1).map((n, i) => ({ id: `le-${n.id}-${ns[i + 1].id}`, source: n.id, target: ns[i + 1].id }));
}

// ── Examples ─────────────────────────────────────────────────────────────────

export function annExample(): Graph {
  const ns = [
    mk("input-1", "input", 0, 160, { params: { shape: "1, 28, 28" } }),
    mk("flatten-2", "flatten", 200, 180),
    mk("dense-3", "linear", 380, 160, { params: { out_features: "128" }, activation: "relu" }),
    mk("dense-4", "linear", 600, 160, { params: { out_features: "3" }, activation: "none" }),
    mk("output-5", "output", 820, 180, { activation: "softmax" }),
  ];
  return { nodes: ns, edges: chain(ns) };
}

export function alexnetExample(classes = 1000): Graph {
  let x = 0;
  const step = () => (x += 150);
  const conv = (id: string, ch: string, k: string, s: string, p: string) =>
    mk(id, "conv2d", step(), 120, { params: { out_channels: ch, kernel_size: k, stride: s, padding: p }, activation: "relu" });
  const pool = (id: string) => mk(id, "maxpool2d", step(), 260, { params: { kernel_size: "3", stride: "2" } });
  const ns = [
    mk("input", "input", step(), 200, { params: { shape: "3, 224, 224" } }),
    conv("conv1", "64", "11", "4", "2"),
    pool("pool1"),
    conv("conv2", "192", "5", "1", "2"),
    pool("pool2"),
    conv("conv3", "384", "3", "1", "1"),
    conv("conv4", "256", "3", "1", "1"),
    conv("conv5", "256", "3", "1", "1"),
    pool("pool3"),
    mk("flatten", "flatten", step(), 320, {}),
    mk("fc6", "linear", step(), 120, { params: { out_features: "4096" }, activation: "relu" }),
    mk("fc7", "linear", step(), 120, { params: { out_features: "4096" }, activation: "relu" }),
    mk("fc8", "linear", step(), 120, { params: { out_features: String(classes) }, activation: "none" }),
    mk("output", "output", step(), 200, { activation: "softmax" }),
  ];
  return { nodes: ns, edges: chain(ns) };
}

export function smallCnnExample(classes = 3): Graph {
  let x = 0;
  const step = () => (x += 160);
  const ns = [
    mk("input", "input", step(), 200, { params: { shape: "1, 28, 28" } }),
    mk("conv1", "conv2d", step(), 120, { params: { out_channels: "8", kernel_size: "3", stride: "1", padding: "1" }, activation: "relu" }),
    mk("pool1", "maxpool2d", step(), 280, { params: { kernel_size: "2", stride: "2" } }),
    mk("conv2", "conv2d", step(), 120, { params: { out_channels: "16", kernel_size: "3", stride: "1", padding: "1" }, activation: "relu" }),
    mk("pool2", "maxpool2d", step(), 280, { params: { kernel_size: "2", stride: "2" } }),
    mk("flatten", "flatten", step(), 320, {}),
    mk("fc1", "linear", step(), 160, { params: { out_features: "32" }, activation: "relu" }),
    mk("fc2", "linear", step(), 160, { params: { out_features: String(classes) }, activation: "none" }),
    mk("output", "output", step(), 200, { activation: "softmax" }),
  ];
  return { nodes: ns, edges: chain(ns) };
}

export function vgg16Example(classes = 1000): Graph {
  // Pure feedforward conv stack (13 conv + 3 linear), so it maps 1:1 onto the
  // graph format and matches torchvision VGG-16 for pretrained-weight loading.
  // Input 224×224 → five /2 pools → 7×7, so a plain flatten replaces torchvision's
  // adaptive-avg-pool (a no-op at 224) and the first Dense sees 512·7·7 = 25088.
  let x = 0;
  const step = () => (x += 110);
  const conv = (id: string, ch: string) =>
    mk(id, "conv2d", step(), 120, { params: { out_channels: ch, kernel_size: "3", stride: "1", padding: "1" }, activation: "relu" });
  const pool = (id: string) => mk(id, "maxpool2d", step(), 260, { params: { kernel_size: "2", stride: "2" } });
  const ns = [
    mk("input", "input", step(), 200, { params: { shape: "3, 224, 224" } }),
    conv("conv1_1", "64"), conv("conv1_2", "64"), pool("pool1"),
    conv("conv2_1", "128"), conv("conv2_2", "128"), pool("pool2"),
    conv("conv3_1", "256"), conv("conv3_2", "256"), conv("conv3_3", "256"), pool("pool3"),
    conv("conv4_1", "512"), conv("conv4_2", "512"), conv("conv4_3", "512"), pool("pool4"),
    conv("conv5_1", "512"), conv("conv5_2", "512"), conv("conv5_3", "512"), pool("pool5"),
    mk("flatten", "flatten", step(), 320, {}),
    mk("fc6", "linear", step(), 120, { params: { out_features: "4096" }, activation: "relu" }),
    mk("fc7", "linear", step(), 120, { params: { out_features: "4096" }, activation: "relu" }),
    mk("fc8", "linear", step(), 120, { params: { out_features: String(classes) }, activation: "none" }),
    mk("output", "output", step(), 200, { activation: "softmax" }),
  ];
  return { nodes: ns, edges: chain(ns) };
}

export const EXAMPLES: { group: string; items: { id: string; label: string; make: () => Graph }[] }[] = [
  { group: "ANN", items: [{ id: "mlp", label: "MLP (28×28 → 3)", make: () => annExample() }] },
  {
    group: "CNN",
    items: [
      { id: "small-cnn", label: "Small CNN (28×28 → 3) · trainable", make: () => smallCnnExample() },
      { id: "alexnet", label: "AlexNet (ImageNet · architecture)", make: () => alexnetExample() },
      { id: "vgg16", label: "VGG-16 (ImageNet · architecture)", make: () => vgg16Example() },
    ],
  },
];

// ── File import / export ─────────────────────────────────────────────────────

export function graphToFile(nodes: TCNode[], edges: { source: string; target: string }[], name: string): NetFile {
  return {
    format: NET_FORMAT,
    version: NET_VERSION,
    name,
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.type,
      position: { x: Math.round(n.position.x), y: Math.round(n.position.y) },
      params: n.data.params,
      activation: n.data.activation,
      bias: n.data.bias,
      customType: n.data.customType,
    })),
    edges: edges.map((e) => ({ source: e.source, target: e.target })),
  };
}

export function fileToGraph(obj: unknown): Graph {
  const o = obj as NetFile;
  if (!o || o.format !== NET_FORMAT || !Array.isArray(o.nodes)) {
    throw new Error("Not a TensorCanvas network file (expected format \"tensorcanvas-net\").");
  }
  const nodes = o.nodes.map((n, i) =>
    mk(n.id ?? `${n.type}-${i}`, n.type, n.position?.x ?? i * 180, n.position?.y ?? 140, {
      params: n.params,
      activation: n.activation as FlowNodeData["activation"],
      bias: n.bias,
      customType: n.customType,
    }),
  );
  const edges: LogicalEdge[] = (o.edges ?? []).map((e) => ({ id: `le-${e.source}-${e.target}`, source: e.source, target: e.target }));
  return { nodes, edges };
}
