// Palette catalog — layer types with a FIXED set of typed attributes.
// Activations and bias are first-class attributes now (no freeform key/values,
// no custom node). Still a 1:1 mapping to backend node types.

export type FieldKind = "number" | "int-list";

export interface FieldSpec {
  key: string; // backend param name
  label: string;
  kind: FieldKind;
  default: string;
  placeholder?: string;
  help?: string;
}

export type NodeCategory = "I/O" | "Layers";

export interface CatalogEntry {
  type: string; // backend layer type
  label: string;
  category: NodeCategory;
  short: string;
  fields: FieldSpec[];
  activation?: boolean; // exposes an activation attribute
  bias?: boolean; // exposes a bias toggle
  neuronCountKey?: string; // which field gives the neuron count (for circle rendering)
}

export const ACTIVATIONS = ["none", "relu", "sigmoid", "tanh", "softmax"] as const;
export type Activation = (typeof ACTIVATIONS)[number];

/** Single-char glyph drawn on a node for each activation. */
export const ACTIVATION_GLYPH: Record<string, string> = {
  none: "",
  relu: "R",
  sigmoid: "σ",
  tanh: "T",
  softmax: "S",
};

/** What we store in each React Flow node's `data`. */
export interface FlowNodeData {
  type: string;
  label: string;
  params: Record<string, string>;
  activation?: Activation;
  bias?: boolean;
  // Populated from the last successful compile (echoed by the backend):
  weights?: number[][] | null;
  biasValues?: number[] | null;
  weightShape?: number[] | null;
  hasBias?: boolean | null;
  // User-set initial values, sent on the next compile:
  initWeights?: number[][] | null;
  initBias?: number[] | null;
  error?: string | null;
  outputShape?: number[] | null;
  [key: string]: unknown;
}

export const CATALOG: CatalogEntry[] = [
  {
    type: "input",
    label: "Input",
    category: "I/O",
    short: "IN",
    fields: [
      {
        key: "shape",
        label: "Shape (per-sample)",
        kind: "int-list",
        default: "1, 28, 28",
        placeholder: "e.g. 1, 28, 28",
        help: "Excludes the batch dimension.",
      },
    ],
  },
  {
    type: "linear",
    label: "Dense",
    category: "Layers",
    short: "FC",
    activation: true,
    bias: true,
    neuronCountKey: "out_features",
    fields: [{ key: "out_features", label: "Neurons", kind: "number", default: "4" }],
  },
  {
    type: "conv2d",
    label: "Conv2D",
    category: "Layers",
    short: "CV",
    activation: true,
    bias: true,
    fields: [
      { key: "out_channels", label: "Out channels", kind: "number", default: "8" },
      { key: "kernel_size", label: "Kernel size", kind: "number", default: "3" },
      { key: "stride", label: "Stride", kind: "number", default: "1" },
      { key: "padding", label: "Padding", kind: "number", default: "0" },
    ],
  },
  {
    type: "maxpool2d",
    label: "MaxPool2D",
    category: "Layers",
    short: "MP",
    fields: [
      { key: "kernel_size", label: "Kernel size", kind: "number", default: "2" },
      { key: "stride", label: "Stride", kind: "number", default: "2" },
    ],
  },
  { type: "flatten", label: "Flatten", category: "Layers", short: "FL", fields: [] },
  { type: "output", label: "Output", category: "I/O", short: "OUT", activation: true, fields: [] },
];

export const CATALOG_BY_TYPE: Record<string, CatalogEntry> = Object.fromEntries(
  CATALOG.map((e) => [e.type, e]),
);

export const CATEGORY_ORDER: NodeCategory[] = ["I/O", "Layers"];

export function defaultDataFor(type: string): FlowNodeData {
  const entry = CATALOG_BY_TYPE[type];
  const params: Record<string, string> = {};
  for (const f of entry?.fields ?? []) params[f.key] = f.default;
  return {
    type,
    label: entry?.label ?? type,
    params,
    activation: entry?.activation ? (type === "linear" ? "relu" : "none") : undefined,
    bias: entry?.bias ? true : undefined,
    weights: null,
    biasValues: null,
    weightShape: null,
    hasBias: null,
    initWeights: null,
    initBias: null,
    error: null,
    outputShape: null,
  };
}

/** Neuron count for a node, if it renders as neurons (Dense). */
export function neuronCount(data: FlowNodeData): number | null {
  const entry = CATALOG_BY_TYPE[data.type];
  if (!entry?.neuronCountKey) return null;
  const n = Number(data.params[entry.neuronCountKey]);
  return Number.isInteger(n) && n > 0 ? n : null;
}
