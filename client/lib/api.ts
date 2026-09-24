// Client for the TensorCanvas backend build endpoint.

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

// ── Wire types (mirror server/schema.py) ─────────────────────────────────────

export interface NodeSpecWire {
  id: string;
  type: string;
  params: Record<string, unknown>;
}

export interface EdgeSpecWire {
  source: string;
  target: string;
}

export interface NetworkSpecWire {
  name?: string | null;
  nodes: NodeSpecWire[];
  edges: EdgeSpecWire[];
}

export interface NodeInfo {
  id: string;
  type: string;
  output_shape: number[];
  num_parameters: number;
  has_bias?: boolean | null;
  weight_shape?: number[] | null;
  weights?: number[][] | null;
  bias_values?: number[] | null;
}

export interface BuildSuccess {
  ok: true;
  model_id: string;
  name: string | null;
  num_parameters: number;
  nodes: NodeInfo[];
}

export interface BuildErrorItem {
  code: string;
  message: string;
  node_id: string | null;
}

export interface BuildFailure {
  ok: false;
  errors: BuildErrorItem[];
}

export type BuildResult =
  | ({ status: number } & BuildSuccess)
  | ({ status: number } & BuildFailure);

/** POST a network spec to the backend. Network/parse errors are surfaced as a
 *  BuildFailure so callers only deal with one shape. */
export async function buildModel(spec: NetworkSpecWire): Promise<BuildResult> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/models/build`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(spec),
    });
    const body = await res.json();
    return { status: res.status, ...body } as BuildResult;
  } catch (err) {
    return {
      status: 0,
      ok: false,
      errors: [
        {
          code: "network",
          message:
            `Could not reach the backend at ${API_BASE}. Is it running? ` +
            `(${err instanceof Error ? err.message : String(err)})`,
          node_id: null,
        },
      ],
    };
  }
}
