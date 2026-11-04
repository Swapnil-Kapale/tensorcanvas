"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useGraph } from "@/lib/store";
import { fetchNodeWeights } from "@/lib/api";
import { patchNodeWeights } from "@/lib/observeApi";
import { ACTIVATIONS, ACTIVATION_GLYPH } from "@/components/build/nodeCatalog";

const MIN_CELL = 14;
const MAX_CELL = 64;
const DEFAULT_CELL = 34; // readable default: numbers fit, cells are big (scrolls)
const CANVAS_LIMIT = 32000; // safe margin under the ~32767px hardware cap
const TEXT_MIN = 26; // cells this big or larger get a white value drawn in them
const ZOOM_STEPS = [14, 18, 22, 28, 34, 40, 48, 56, 64];
const PAD = 32;

function fmt(v: number, cell: number): string {
  if (cell >= 46) return v.toFixed(2);
  if (cell >= 30) return v.toFixed(2).replace(/^(-?)0\./, "$1.");
  return v.toFixed(1);
}

function stepZoom(cur: number, dir: number, hardMax: number): number {
  const steps = ZOOM_STEPS.filter((s) => s >= MIN_CELL && s <= hardMax);
  if (steps.length === 0) return Math.max(MIN_CELL, Math.min(hardMax, cur));
  let idx = 0;
  let best = Infinity;
  steps.forEach((s, i) => {
    const d = Math.abs(s - cur);
    if (d < best) {
      best = d;
      idx = i;
    }
  });
  return steps[Math.max(0, Math.min(steps.length - 1, idx + dir))];
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  if (r <= 0.5) {
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    return;
  }
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function StatTile({ label, value, mono, accent }: { label: string; value: string; mono?: boolean; accent?: boolean }) {
  return (
    <div className="flex flex-col rounded-lg bg-neutral-800/50 px-2.5 py-1 ring-1 ring-inset ring-neutral-800">
      <span className="text-[10px] uppercase tracking-wide text-neutral-500">{label}</span>
      <span
        className={[
          "text-xs font-semibold",
          mono ? "font-mono" : "",
          accent ? "text-sky-300" : "text-neutral-200",
        ].join(" ")}
      >
        {value}
      </span>
    </div>
  );
}

export function LayerEditor({ nodeId, mode = "initial" }: { nodeId: string; mode?: "initial" | "live" }) {
  const router = useRouter();
  const node = useGraph((s) => s.nodes.find((n) => n.id === nodeId));
  const lastModelId = useGraph((s) => s.lastModelId);
  const updateNodeData = useGraph((s) => s.updateNodeData);
  const [saveErr, setSaveErr] = useState<string | null>(null);

  const seedW = node?.data.initWeights ?? node?.data.weights ?? null;
  const [matrix, setMatrix] = useState<number[][] | null>(() => (seedW ? seedW.map((r) => [...r]) : null));
  const [bias, setBias] = useState<number[] | null>(() => node?.data.initBias ?? node?.data.biasValues ?? null);
  const [perAct, setPerAct] = useState<(string | null)[]>(() =>
    node?.data.neuronActivations ? [...node.data.neuronActivations] : [],
  );
  const [layerAct, setLayerAct] = useState<string>(node?.data.activation ?? "none");
  const [useBias, setUseBias] = useState<boolean>(node?.data.bias ?? true);
  const [sel, setSel] = useState(0);
  const [selCol, setSelCol] = useState<number | null>(null);
  const [fetchErr, setFetchErr] = useState<string | null>(null);
  const [avail, setAvail] = useState({ w: 900, h: 600 });
  const [zoom, setZoom] = useState<number | null>(null);
  const [hover, setHover] = useState<{ i: number; j: number; v: number } | null>(null);
  const [scroll, setScroll] = useState({ left: 0, top: 0 });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const selRowRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (matrix || !node || !lastModelId) return;
    let cancelled = false;
    fetchNodeWeights(lastModelId, nodeId)
      .then((w) => {
        if (cancelled) return;
        setMatrix(w.weights.map((r) => [...r]));
        setBias((prev) => prev ?? w.bias_values ?? null);
        setPerAct((prev) => (prev.length === w.weights.length ? prev : new Array(w.weights.length).fill(null)));
      })
      .catch((e) => !cancelled && setFetchErr(e?.message ?? String(e)));
    return () => {
      cancelled = true;
    };
  }, [matrix, node, lastModelId, nodeId]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setAvail({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const out = matrix?.length ?? 0;
  const inp = matrix?.[0]?.length ?? 0;

  const stats = useMemo(() => {
    let min = Infinity,
      max = -Infinity,
      sum = 0,
      absMax = 1e-9,
      n = 0;
    for (const r of matrix ?? [])
      for (const v of r) {
        min = Math.min(min, v);
        max = Math.max(max, v);
        sum += v;
        absMax = Math.max(absMax, Math.abs(v));
        n++;
      }
    return { min, max, mean: sum / Math.max(n, 1), absMax, n, params: n + (useBias ? out : 0) };
  }, [matrix, useBias, out]);

  const hardMax = useMemo(
    () => Math.max(MIN_CELL, Math.min(MAX_CELL, Math.floor(CANVAS_LIMIT / Math.max(inp, out, 1)))),
    [inp, out],
  );
  const cell = useMemo(() => {
    const fit = Math.floor(Math.min((avail.w - PAD) / Math.max(inp, 1), (avail.h - PAD) / Math.max(out, 1)));
    // Default to a readable size (numbers visible); small layers still grow to fill.
    const base = zoom ?? Math.max(fit, DEFAULT_CELL);
    return Math.max(MIN_CELL, Math.min(hardMax, base));
  }, [avail, inp, out, zoom, hardMax]);

  // Content origin inside the viewport: centered when it fits, else scrolled.
  const ox = useMemo(
    () => (inp * cell < avail.w ? Math.floor((avail.w - inp * cell) / 2) : -scroll.left),
    [inp, cell, avail.w, scroll.left],
  );
  const oy = useMemo(
    () => (out * cell < avail.h ? Math.floor((avail.h - out * cell) / 2) : -scroll.top),
    [out, cell, avail.h, scroll.top],
  );

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !matrix || out === 0) return;
    const el = wrapRef.current;
    const vw = Math.max(1, el ? el.clientWidth : Math.floor(avail.w));
    const vh = Math.max(1, el ? el.clientHeight : Math.floor(avail.h));
    c.width = vw;
    c.height = vh;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const absMax = stats.absMax;
    ctx.clearRect(0, 0, vw, vh);
    const showText = cell >= TEXT_MIN;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = `${Math.max(9, Math.round(cell * 0.3))}px ui-monospace, SFMono-Regular, monospace`;
    const r = Math.min(6, Math.floor(cell * 0.16));
    // Virtualized: only draw the cells visible in the viewport.
    const j0 = Math.max(0, Math.floor(-ox / cell));
    const j1 = Math.min(inp, Math.ceil((vw - ox) / cell));
    const i0 = Math.max(0, Math.floor(-oy / cell));
    const i1 = Math.min(out, Math.ceil((vh - oy) / cell));
    for (let i = i0; i < i1; i++) {
      for (let j = j0; j < j1; j++) {
        const v = matrix[i][j];
        const x = ox + j * cell;
        const y = oy + i * cell;
        ctx.fillStyle = `rgba(56,189,248,${0.06 + Math.min(1, Math.abs(v) / absMax) * 0.94})`;
        roundRect(ctx, x + 1, y + 1, cell - 2, cell - 2, r);
        ctx.fill();
        if (showText) {
          ctx.shadowColor = "rgba(0,0,0,0.6)";
          ctx.shadowBlur = 2;
          ctx.fillStyle = "rgba(255,255,255,0.96)";
          ctx.fillText(fmt(v, cell), x + cell / 2, y + cell / 2 + 0.5);
          ctx.shadowBlur = 0;
        }
      }
    }
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(56,189,248,0.9)";
    ctx.strokeRect(ox + 1, oy + sel * cell + 1, inp * cell - 2, cell - 2);
    if (selCol != null) {
      ctx.strokeStyle = "rgba(251,191,36,0.95)";
      ctx.strokeRect(ox + selCol * cell + 1, oy + sel * cell + 1, cell - 2, cell - 2);
    }
  }, [matrix, sel, selCol, out, inp, cell, stats.absMax, ox, oy, avail.w, avail.h]);

  useEffect(() => {
    selRowRef.current?.scrollIntoView({ block: "nearest" });
  }, [sel, selCol]);

  if (!node) {
    return (
      <Frame onBack={() => router.push("/")}>
        <p className="p-10 text-neutral-400">Node not found. It may have been removed.</p>
      </Frame>
    );
  }
  if (!matrix) {
    const msg = fetchErr ?? (lastModelId ? "Loading…" : "Compile the network first to load this layer's weights.");
    return (
      <Frame onBack={() => router.push("/")} title={node.data.label}>
        <p className="p-10 text-neutral-400">{msg}</p>
      </Frame>
    );
  }

  const setCell = (i: number, j: number, raw: string) => {
    const v = Number(raw);
    if (Number.isNaN(v)) return;
    setMatrix((m) => {
      if (!m) return m;
      const next = m.map((row) => [...row]);
      next[i][j] = v;
      return next;
    });
  };
  const setBiasAt = (i: number, raw: string) => {
    const v = Number(raw);
    if (Number.isNaN(v)) return;
    setBias((b) => {
      const next = (b ?? new Array(out).fill(0)).slice();
      next[i] = v;
      return next;
    });
  };
  const setNeuronAct = (i: number, value: string) =>
    setPerAct((p) => {
      const next = p.length === out ? [...p] : new Array(out).fill(null);
      next[i] = value === "none" ? null : value;
      return next;
    });
  const pick = (i: number) => {
    setSel(Math.min(out - 1, Math.max(0, i)));
    setSelCol(null);
  };

  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const j = Math.floor((e.clientX - rect.left - ox) / cell);
    const i = Math.floor((e.clientY - rect.top - oy) / cell);
    if (i >= 0 && i < out) {
      setSel(i);
      setSelCol(j >= 0 && j < inp ? j : null);
    }
  };
  const onHover = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const j = Math.floor((e.clientX - rect.left - ox) / cell);
    const i = Math.floor((e.clientY - rect.top - oy) / cell);
    if (i >= 0 && i < out && j >= 0 && j < inp) setHover({ i, j, v: matrix[i][j] });
    else setHover(null);
  };

  const save = async () => {
    if (mode === "live") {
      if (!lastModelId) return;
      try {
        const b = useBias ? bias ?? new Array(out).fill(0) : null;
        await patchNodeWeights(lastModelId, nodeId, { weights: matrix, bias: b });
        updateNodeData(nodeId, { weights: matrix, biasValues: b });
        router.push("/");
      } catch (e) {
        setSaveErr(e instanceof Error ? e.message : String(e));
      }
      return;
    }
    updateNodeData(nodeId, {
      initWeights: matrix,
      initBias: useBias ? bias ?? new Array(out).fill(0) : null,
      neuronActivations: perAct.length === out && perAct.some((a) => a) ? perAct : null,
      activation: layerAct as never,
      bias: useBias,
    });
    router.push("/");
  };

  const micro = "text-[11px] font-semibold uppercase tracking-wider text-neutral-500";

  return (
    <div className="flex h-full min-h-0 flex-col bg-neutral-950 text-neutral-100 antialiased">
      {/* HEADER */}
      <header className="border-b border-neutral-800">
        <div className="flex items-center gap-4 px-6 py-3">
          <button
            onClick={() => router.push("/")}
            className="flex h-9 items-center gap-1.5 rounded-lg border border-neutral-700/80 px-3 text-sm text-neutral-300 transition hover:border-neutral-600 hover:bg-neutral-800/60"
          >
            <span aria-hidden>←</span> Back
          </button>
          <div className="h-6 w-px bg-neutral-800" />
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-sky-500/15 font-mono text-sm text-sky-300 ring-1 ring-inset ring-sky-500/30">
            W
          </div>
          <div className="flex items-baseline gap-2.5">
            <h1 className="text-base font-semibold tracking-tight text-neutral-50">{node.data.label}</h1>
            <span className="rounded-md bg-neutral-800 px-2 py-0.5 font-mono text-xs font-medium text-neutral-400">
              {node.id}
            </span>
            <span className="font-mono text-xs text-neutral-500">
              {out} × {inp}
            </span>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {mode === "live" && (
              <span className="rounded-md bg-violet-500/15 px-2 py-1 text-[11px] text-violet-200 ring-1 ring-inset ring-violet-500/30">
                live edit · applies to the trained model
              </span>
            )}
            {saveErr && <span className="max-w-[260px] truncate text-xs text-red-300">{saveErr}</span>}
            <button
              onClick={() => router.push("/")}
              className="h-9 rounded-lg border border-neutral-700/80 px-4 text-sm text-neutral-300 transition hover:bg-neutral-800/60"
            >
              Discard
            </button>
            <button
              onClick={save}
              className="h-9 rounded-lg bg-sky-500 px-4 text-sm font-semibold text-neutral-950 shadow-lg shadow-sky-950/50 transition hover:bg-sky-400 active:bg-sky-500"
            >
              {mode === "live" ? "Apply (live)" : "Apply changes"}
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2 overflow-x-auto border-t border-neutral-800/60 bg-neutral-900/40 px-6 py-2.5">
          <label className="flex items-center gap-2 rounded-lg bg-violet-500/10 px-2.5 py-1.5 text-xs ring-1 ring-inset ring-violet-500/25">
            <span className="font-medium text-violet-300">activation</span>
            <select
              value={layerAct}
              onChange={(e) => setLayerAct(e.target.value)}
              className="rounded-md bg-violet-500/15 px-2 py-0.5 text-xs font-semibold text-violet-100 outline-none focus:ring-1 focus:ring-violet-400"
            >
              {ACTIVATIONS.map((a) => (
                <option key={a} value={a} className="bg-neutral-900">
                  {a} {ACTIVATION_GLYPH[a] ? `(${ACTIVATION_GLYPH[a]})` : ""}
                </option>
              ))}
            </select>
          </label>
          <label
            className={[
              "flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-1.5 text-xs ring-1 ring-inset transition",
              useBias
                ? "bg-amber-500/10 text-amber-200 ring-amber-500/25"
                : "bg-neutral-800/60 text-neutral-400 ring-neutral-700",
            ].join(" ")}
          >
            <input
              type="checkbox"
              checked={useBias}
              onChange={(e) => setUseBias(e.target.checked)}
              className="h-3.5 w-3.5 accent-amber-400"
            />
            bias
          </label>
          <div className="mx-1 h-5 w-px bg-neutral-800" />
          <StatTile label="params" value={stats.params.toLocaleString()} />
          <StatTile label="min" value={stats.min.toFixed(3)} mono />
          <StatTile label="max" value={stats.max.toFixed(3)} mono />
          <StatTile label="mean" value={stats.mean.toFixed(3)} mono />
          <StatTile label="|max|" value={stats.absMax.toFixed(3)} mono accent />
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* HEATMAP */}
        <section className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="flex items-center gap-3 px-6 pt-4 pb-2">
            <p className="text-xs text-neutral-500">Click a cell to focus a neuron / weight · blue = |weight|</p>
            <div className="ml-auto flex items-center gap-2">
              <span className="min-w-[180px] rounded-md bg-neutral-800/70 px-2.5 py-1 text-right font-mono text-xs text-neutral-300 ring-1 ring-inset ring-neutral-700">
                {hover ? `w[${hover.i}][${hover.j}] = ${hover.v.toFixed(4)}` : "hover a cell"}
              </span>
              <div className="flex items-center rounded-lg ring-1 ring-inset ring-neutral-700">
                <button
                  onClick={() => setZoom((z) => stepZoom(z ?? cell, -1, hardMax))}
                  className="px-2.5 py-1 text-neutral-300 hover:bg-neutral-800"
                >
                  −
                </button>
                <span className="w-14 text-center font-mono text-xs text-neutral-400">{cell}px</span>
                <button
                  onClick={() => setZoom((z) => stepZoom(z ?? cell, +1, hardMax))}
                  className="px-2.5 py-1 text-neutral-300 hover:bg-neutral-800"
                >
                  +
                </button>
              </div>
              <button
                onClick={() => setZoom(null)}
                title="Reset to a readable default size"
                className="rounded-lg px-2.5 py-1 text-xs text-neutral-400 ring-1 ring-inset ring-neutral-700 hover:bg-neutral-800"
              >
                Reset
              </button>
            </div>
          </div>
          <div
            ref={wrapRef}
            onScroll={(e) => setScroll({ left: e.currentTarget.scrollLeft, top: e.currentTarget.scrollTop })}
            className="relative mx-6 min-h-0 flex-1 overflow-auto rounded-2xl border border-neutral-800 bg-[radial-gradient(circle_at_30%_20%,#171717,#0a0a0a)]"
          >
            <div aria-hidden style={{ width: inp * cell, height: out * cell }} />
            <canvas
              ref={canvasRef}
              onClick={onCanvasClick}
              onMouseMove={onHover}
              onMouseLeave={() => setHover(null)}
              style={{ position: "absolute", top: 0, left: 0, transform: `translate(${scroll.left}px, ${scroll.top}px)` }}
              className="cursor-crosshair"
            />
          </div>
          <div className="flex items-center gap-2 px-6 py-3">
            <span className="text-xs text-neutral-500">|weight|</span>
            <span className="font-mono text-[10px] text-neutral-500">0</span>
            <div
              className="h-2.5 w-40 rounded-full ring-1 ring-inset ring-neutral-700"
              style={{ background: "linear-gradient(90deg, rgba(56,189,248,0.06), rgba(56,189,248,1))" }}
            />
            <span className="font-mono text-[10px] text-neutral-500">{stats.absMax.toFixed(2)}</span>
            <span className="ml-3 font-mono text-[10px] text-neutral-500">μ {stats.mean.toFixed(3)}</span>
          </div>
        </section>

        {/* SIDEBAR */}
        <aside className="flex w-[440px] shrink-0 flex-col border-l border-neutral-800 bg-neutral-900/40 lg:w-[380px] xl:w-[440px]">
          <div className="border-b border-neutral-800 p-5">
            <div className="flex items-center justify-between">
              <span className={micro}>Neuron</span>
              <span className="font-mono text-xs text-neutral-500">
                {sel + 1} / {out}
              </span>
            </div>
            <div className="mt-3 flex items-center gap-2">
              <button
                onClick={() => pick(sel - 1)}
                disabled={sel === 0}
                className="h-9 w-9 rounded-lg text-neutral-300 ring-1 ring-inset ring-neutral-700 hover:bg-neutral-800 disabled:opacity-30"
              >
                ‹
              </button>
              <input
                type="number"
                min={0}
                max={out - 1}
                value={sel}
                onChange={(e) => pick(Number(e.target.value) || 0)}
                className="h-9 w-full rounded-lg border border-neutral-700 bg-neutral-800/80 px-3 text-center font-mono text-base font-semibold text-neutral-100 outline-none focus:border-sky-500"
              />
              <button
                onClick={() => pick(sel + 1)}
                disabled={sel === out - 1}
                className="h-9 w-9 rounded-lg text-neutral-300 ring-1 ring-inset ring-neutral-700 hover:bg-neutral-800 disabled:opacity-30"
              >
                ›
              </button>
            </div>
            <div className="mt-3 flex max-h-28 flex-wrap gap-1 overflow-auto pr-1">
              {Array.from({ length: out }).map((_, i) => (
                <button
                  key={i}
                  onClick={() => pick(i)}
                  title={`neuron ${i}`}
                  className={
                    i === sel
                      ? "h-5 w-5 rounded-md bg-sky-400 ring-2 ring-sky-300/60"
                      : "h-5 w-5 rounded-md bg-sky-400/15 ring-1 ring-inset ring-sky-400/30 transition hover:bg-sky-400/30"
                  }
                />
              ))}
            </div>
          </div>

          <div className="space-y-3 border-b border-neutral-800 p-5">
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm text-neutral-400">activation</span>
              <select
                value={perAct[sel] ?? "none"}
                onChange={(e) => setNeuronAct(sel, e.target.value)}
                className="rounded-lg border border-violet-700/40 bg-violet-500/10 px-3 py-2 text-sm font-medium text-violet-100 outline-none focus:border-violet-500"
              >
                {ACTIVATIONS.map((a) => (
                  <option key={a} value={a} className="bg-neutral-900">
                    {a}
                  </option>
                ))}
              </select>
            </div>
            {useBias && (
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm text-neutral-400">bias</span>
                <input
                  value={bias?.[sel] ?? 0}
                  onChange={(e) => setBiasAt(sel, e.target.value)}
                  className="w-32 rounded-lg border border-amber-700/40 bg-amber-500/10 px-3 py-2 text-right font-mono text-sm text-amber-200 outline-none focus:border-amber-500"
                />
              </div>
            )}
          </div>

          <div className="flex min-h-0 flex-1 flex-col p-5">
            <div className="mb-3 flex items-center justify-between">
              <span className={micro}>Incoming weights</span>
              <span className="font-mono text-xs text-neutral-500">{inp}</span>
            </div>
            <div className="min-h-0 flex-1 space-y-1 overflow-auto">
              {matrix[sel].map((v, j) => (
                <div
                  key={j}
                  ref={j === selCol ? selRowRef : undefined}
                  className={
                    j === selCol
                      ? "relative flex items-center gap-2.5 rounded-lg bg-amber-500/10 px-2 py-1 ring-1 ring-inset ring-amber-500/40"
                      : "relative flex items-center gap-2.5 rounded-lg px-2 py-1 transition hover:bg-neutral-800/40"
                  }
                >
                  <span className="w-9 shrink-0 font-mono text-[11px] text-neutral-600">w{j}</span>
                  <div className="relative flex-1">
                    <div
                      className="pointer-events-none absolute inset-y-0 left-0 rounded-md bg-sky-500/15"
                      style={{ width: `${Math.min(100, (Math.abs(v) / stats.absMax) * 100)}%` }}
                    />
                    <input
                      value={Number.isInteger(v) ? v : Number(v.toFixed(5))}
                      onChange={(e) => setCell(sel, j, e.target.value)}
                      className="relative w-full rounded-md border border-neutral-700/60 bg-transparent px-2.5 py-1.5 text-right font-mono text-sm text-neutral-100 outline-none focus:border-sky-500 focus:bg-neutral-800/70"
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

function Frame({ onBack, title, children }: { onBack: () => void; title?: string; children: React.ReactNode }) {
  return (
    <div className="flex h-full min-h-0 flex-col bg-neutral-950 text-neutral-100">
      <header className="flex items-center gap-4 border-b border-neutral-800 px-6 py-4">
        <button
          onClick={onBack}
          className="rounded-lg border border-neutral-700 px-4 py-2 text-sm text-neutral-300 transition hover:bg-neutral-800"
        >
          ← Back
        </button>
        <h1 className="text-base font-semibold">
          Edit layer{title ? <span className="font-normal text-neutral-400"> · {title}</span> : null}
        </h1>
      </header>
      <div className="flex min-h-0 flex-1 flex-col">{children}</div>
    </div>
  );
}
