// Palette catalog — layer types with fixed typed attributes, plus a free-form
// Custom node. 1:1 with backend node types.

export type FieldKind = "number" | "int-list";

export interface FieldSpec {
  key: string; // backend param name
  label: string;
  kind: FieldKind;
  default: string;
  placeholder?: string;
  help?: string;
}

export type NodeCategory = "I/O" | "Layers" | "Other";

export interface CatalogEntry {
  type: string; // backend layer type ("custom" = free-form)
  label: string;
  category: NodeCategory;
  short: string;
  fields: FieldSpec[];
  activation?: boolean;
  bias?: boolean;
  neuronCountKey?: string; // which field gives the neuron count (Dense)
  custom?: boolean;
}

export const ACTIVATIONS = ["none", "relu", "sigmoid", "tanh", "softmax"] as const;
export type Activation = (typeof ACTIVATIONS)[number];

export const ACTIVATION_GLYPH: Record<string, string> = {
  none: "",
  relu: "R",
  sigmoid: "σ",
  tanh: "T",
  softmax: "S",
};

export interface ParamRow {
  key: string;
  value: string;
}

export interface FlowNodeData {
  type: string;
  label: string;
  params: Record<string, string>;
  activation?: Activation;
  bias?: boolean;
  // Custom node:
  customType?: string;
  extra?: ParamRow[];
  // Per-neuron activation overrides (index-aligned, length = out_features):
  neuronActivations?: (string | null)[] | null;
  // Populated from the last successful compile:
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
  { type: "custom", label: "Custom node", category: "Other", short: "{ }", custom: true, fields: [] },
];

export const CATALOG_BY_TYPE: Record<string, CatalogEntry> = Object.fromEntries(
  CATALOG.map((e) => [e.type, e]),
);

/** Right-click menu entries (includes quick presets that are not their own type). */
export interface PaletteItem {
  label: string;
  type: string;
  short: string;
  category: NodeCategory;
  preset?: Partial<FlowNodeData>;
}

export const PALETTE_ITEMS: PaletteItem[] = [
  { label: "Input", type: "input", short: "IN", category: "I/O" },
  { label: "Neuron", type: "linear", short: "●", category: "Layers", preset: { params: { out_features: "1" } } },
  { label: "Dense", type: "linear", short: "FC", category: "Layers" },
  { label: "Conv2D", type: "conv2d", short: "CV", category: "Layers" },
  { label: "MaxPool2D", type: "maxpool2d", short: "MP", category: "Layers" },
  { label: "Flatten", type: "flatten", short: "FL", category: "Layers" },
  { label: "Output", type: "output", short: "OUT", category: "I/O" },
  { label: "Custom node", type: "custom", short: "{ }", category: "Other" },
];

export const CATEGORY_ORDER: NodeCategory[] = ["I/O", "Layers", "Other"];

export function defaultDataFor(type: string, preset?: Partial<FlowNodeData>): FlowNodeData {
  const entry = CATALOG_BY_TYPE[type];
  const params: Record<string, string> = {};
  for (const f of entry?.fields ?? []) params[f.key] = f.default;
  const base: FlowNodeData = {
    type,
    label: entry?.label ?? type,
    params,
    activation: entry?.activation ? (type === "linear" ? "relu" : "none") : undefined,
    bias: entry?.bias ? true : undefined,
    customType: entry?.custom ? "" : undefined,
    extra: entry?.custom ? [] : undefined,
    neuronActivations: null,
    weights: null,
    biasValues: null,
    weightShape: null,
    hasBias: null,
    initWeights: null,
    initBias: null,
    error: null,
    outputShape: null,
  };
  if (preset) {
    return { ...base, ...preset, params: { ...base.params, ...(preset.params ?? {}) } };
  }
  return base;
}

/** Neuron count for a Dense node, else null. */
export function neuronCount(data: FlowNodeData): number | null {
  const entry = CATALOG_BY_TYPE[data.type];
  if (!entry?.neuronCountKey) return null;
  const n = Number(data.params[entry.neuronCountKey]);
  return Number.isInteger(n) && n > 0 ? n : null;
}

/** Whether a Dense node renders as individual neurons at the given threshold. */
export function isExpanded(data: FlowNodeData, threshold: number): boolean {
  const n = neuronCount(data);
  return n != null && n <= threshold;
}
