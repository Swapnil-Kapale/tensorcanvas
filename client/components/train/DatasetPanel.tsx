"use client";

import { useEffect, useRef, useState } from "react";
import { builtinDataset, uploadDataset, type DatasetInfo, type ModelIO } from "@/lib/trainApi";
import { DecisionBoundary } from "./DecisionBoundary";

export function DatasetPanel({
  modelIO,
  dataset,
  validation,
  disabled,
  onDataset,
}: {
  modelIO: ModelIO | null;
  dataset: DatasetInfo | null;
  validation: { ok: boolean; detail?: string } | null;
  disabled: boolean;
  onDataset: (d: DatasetInfo) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setBusy(true);
    setErr(null);
    try {
      onDataset(await uploadDataset(file));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const pickBuiltin = async (name: string) => {
    setBusy(true);
    setErr(null);
    try {
      onDataset(await builtinDataset(name, { n: 1000, classes: Math.max(2, modelIO?.out_dim ?? 2) }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4 border-b border-[var(--stroke)] p-4">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--faint)]">Dataset</div>

      {/* upload dropzone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
        }}
        onClick={() => inputRef.current?.click()}
        className={[
          "cursor-pointer rounded-xl border-2 border-dashed p-4 text-center text-xs transition",
          drag ? "border-sky-500 bg-sky-500/10 text-sky-200" : "border-[var(--stroke)] text-[var(--muted)] hover:border-[var(--stroke-strong)]",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.json,.npy,.npz,.zip"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        <div className="font-medium text-[var(--muted)]">{busy ? "Uploading…" : "Drop a dataset or click to upload"}</div>
        <div className="mt-1 text-[10px] text-[var(--faint)]">CSV · JSON · NPY/NPZ · image ZIP (class folders)</div>
      </div>

      {modelIO?.builtins?.length ? (
        <div>
          <div className="mb-1 text-[10px] text-[var(--faint)]">or a built-in (2-D):</div>
          <div className="flex flex-wrap gap-1.5">
            {modelIO.builtins.map((b) => (
              <button
                key={b.id}
                title={b.blurb}
                disabled={busy}
                onClick={() => pickBuiltin(b.id)}
                className="rounded-lg border border-[var(--stroke)] px-2.5 py-1 text-xs text-[var(--muted)] hover:bg-[var(--glass-hover)] disabled:opacity-40"
              >
                {b.id}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {err && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-2 text-[11px] text-red-300">{err}</div>}

      {dataset && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-mono text-[var(--muted)]">{dataset.format}</span>
            <span className="text-[var(--faint)]">
              {dataset.n_samples} samples · {dataset.task}
              {dataset.n_classes ? ` · ${dataset.n_classes} cls` : ""}
            </span>
          </div>
          {dataset.scatter ? (
            <DecisionBoundary boundary={null} scatter={dataset.scatter} size={240} />
          ) : dataset.thumbs ? (
            <div className="flex flex-wrap gap-1">
              {dataset.thumbs.images.slice(0, 12).map((img, i) => (
                <Thumb key={i} data={img} size={dataset.thumbs!.size} />
              ))}
            </div>
          ) : (
            <div className="rounded-lg bg-black/25 p-2 text-[11px] text-[var(--faint)]">
              feature shape [{(dataset.feature_shape ?? []).join(", ")}]
            </div>
          )}
          {validation && (
            <div
              className={[
                "rounded-lg p-2 text-[11px]",
                validation.ok
                  ? "border border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                  : "border border-red-500/40 bg-red-500/10 text-red-300",
              ].join(" ")}
            >
              {validation.ok ? "✓ matches this network" : `✗ ${validation.detail}`}
            </div>
          )}
        </div>
      )}
      {disabled && <div className="text-[10px] text-[var(--faint)]">Stop the current run to change the dataset.</div>}
    </div>
  );
}

function Thumb({ data, size }: { data: number[]; size: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(size, size);
    for (let i = 0; i < size * size; i++) {
      const v = Math.round((data[i] ?? 0) * 255);
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data, size]);
  return <canvas ref={ref} width={size} height={size} className="h-7 w-7 rounded border border-[var(--stroke)]" />;
}
