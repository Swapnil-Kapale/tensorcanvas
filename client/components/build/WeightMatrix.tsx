"use client";

import { useMemo } from "react";
import type { FlowNodeData } from "./nodeCatalog";

export interface WeightMatrixProps {
  data: FlowNodeData;
  onSetInit: (initWeights: number[][] | null, initBias: number[] | null) => void;
}

const EDIT_MAX = 16; // editable inline only when out<=16 && in<=16
const HEAT_ROWS = 8;
const HEAT_COLS = 16;

function fmt(n: number): string {
  return Math.abs(n) < 1e-4 && n !== 0 ? n.toExponential(1) : n.toFixed(3);
}

export function WeightMatrix({ data, onSetInit }: WeightMatrixProps) {
  const shape = data.weightShape;
  const weights = data.weights ?? null;

  const [out, inp] = shape ?? [0, 0];
  const editable = Boolean(shape && weights && out <= EDIT_MAX && inp <= EDIT_MAX);

  // Seed for editing: user's init if set, else the last compiled weights.
  const seedW = data.initWeights ?? weights;
  const seedB = data.initBias ?? data.biasValues ?? null;

  const max = useMemo(() => {
    if (!weights) return 1;
    let m = 1e-9;
    for (const row of weights) for (const v of row) m = Math.max(m, Math.abs(v));
    return m;
  }, [weights]);

  if (!shape) {
    return <p className="text-[11px] text-[var(--faint)]">Compile to view weights.</p>;
  }

  if (editable && seedW) {
    const setCell = (r: number, c: number, raw: string) => {
      const v = Number(raw);
      if (Number.isNaN(v)) return;
      const w = seedW.map((row) => [...row]);
      w[r][c] = v;
      onSetInit(w, seedB);
    };
    const setBias = (r: number, raw: string) => {
      const v = Number(raw);
      if (Number.isNaN(v)) return;
      const b = (seedB ?? new Array(out).fill(0)).slice();
      b[r] = v;
      onSetInit(seedW.map((row) => [...row]), b);
    };
    return (
      <div>
        <div className="mb-1 flex items-center justify-between">
          <span className="font-mono text-[11px] text-[var(--faint)]">
            W [{out}×{inp}]{data.hasBias ? " + b" : ""}
          </span>
          {(data.initWeights || data.initBias) && (
            <button
              className="text-[11px] text-sky-400 hover:text-sky-300"
              onClick={() => onSetInit(null, null)}
            >
              reset to random
            </button>
          )}
        </div>
        <div className="space-y-1">
          {seedW.map((row, r) => (
            <div key={r} className="flex items-center gap-1">
              <span className="w-6 shrink-0 font-mono text-[10px] text-[var(--faint)]">n{r}</span>
              {row.map((v, c) => (
                <input
                  key={c}
                  value={fmt(v)}
                  onChange={(e) => setCell(r, c, e.target.value)}
                  className="w-12 rounded border border-[var(--stroke)] bg-black/30 px-1 py-0.5 text-right font-mono text-[10px] text-[var(--text)] outline-none focus:border-[var(--accent)]"
                />
              ))}
              {data.hasBias && (
                <input
                  value={fmt((seedB ?? [])[r] ?? 0)}
                  onChange={(e) => setBias(r, e.target.value)}
                  title="bias"
                  className="w-12 rounded border border-amber-700/60 bg-black/30 px-1 py-0.5 text-right font-mono text-[10px] text-amber-200 outline-none focus:border-amber-500"
                />
              )}
            </div>
          ))}
        </div>
        <p className="mt-1 text-[10px] text-[var(--faint)]">
          Edited values are sent as initial weights on the next compile.
        </p>
      </div>
    );
  }

  // Read-only heatmap for medium layers.
  if (weights) {
    const rows = Math.min(out, HEAT_ROWS);
    const cols = Math.min(inp, HEAT_COLS);
    return (
      <div>
        <div className="mb-1 font-mono text-[11px] text-[var(--faint)]">
          W [{out}×{inp}] — read-only
        </div>
        <div className="space-y-0.5">
          {Array.from({ length: rows }).map((_, r) => (
            <div key={r} className="flex gap-0.5">
              {Array.from({ length: cols }).map((_, c) => {
                const v = weights[r][c];
                return (
                  <div
                    key={c}
                    title={fmt(v)}
                    className="h-3.5 w-3.5 rounded-[2px]"
                    style={{ background: `rgba(56,189,248,${Math.min(1, Math.abs(v) / max)})` }}
                  />
                );
              })}
              {inp > cols && <span className="pl-1 text-[9px] text-[var(--faint)]">+{inp - cols}</span>}
            </div>
          ))}
        </div>
        {out > rows && <div className="mt-0.5 text-[9px] text-[var(--faint)]">+{out - rows} more rows</div>}
      </div>
    );
  }

  return (
    <p className="font-mono text-[11px] text-[var(--faint)]">
      W [{out}×{inp}] — too large to show inline.
    </p>
  );
}
