"use client";

import { useEffect, useRef, useState } from "react";
import { useObserve } from "@/lib/observeStore";
import { uploadDataset } from "@/lib/trainApi";
import { listPretrained, loadPretrained } from "@/lib/observeApi";

export function InputPicker({ modelId, inputShape }: { modelId: string; inputShape: number[] }) {
  const run = useObserve((s) => s.run);
  const busy = useObserve((s) => s.busy);
  const dim = inputShape.reduce((a, b) => a * b, 1);
  const small = inputShape.length === 1 && dim <= 16;

  const [vals, setVals] = useState<number[]>(() => new Array(small ? dim : 0).fill(0));
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [nSamples, setNSamples] = useState(0);
  const [index, setIndex] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [pre, setPre] = useState<string | null>(null);
  const [archs, setArchs] = useState<string[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLInputElement>(null);
  const isImage = inputShape.length === 3;

  useEffect(() => {
    if (!isImage) return;
    let alive = true;
    listPretrained()
      .then((a) => { if (alive && a.length) setArchs(a); })
      .catch(() => {});
    return () => { alive = false; };
  }, [isImage]);

  const onPretrained = async (arch: string) => {
    const name = arch.toUpperCase();
    setPre(`Loading pretrained ${name}… (first run downloads the weights — can take a while)`);
    setErr(null);
    try {
      const r = await loadPretrained(modelId, arch);
      setPre(`Pretrained ${name} loaded · ${r.classes} ImageNet classes — upload an image to classify`);
    } catch (e) {
      setPre(null);
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  const onImage = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => run(modelId, { kind: "image", data: String(reader.result) });
    reader.readAsDataURL(file);
  };

  const field = "w-full rounded-md border border-neutral-700 bg-neutral-800/80 px-2 py-1 text-right font-mono text-xs text-neutral-100 outline-none focus:border-violet-500";
  const btn = "rounded-lg px-3 py-1.5 text-sm font-semibold transition disabled:opacity-40";

  const onUpload = async (file: File) => {
    setErr(null);
    try {
      const info = await uploadDataset(file);
      setDatasetId(info.dataset_id);
      setNSamples(info.n_samples);
      setIndex(0);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="space-y-3 border-b border-neutral-800 p-4">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-neutral-500">
        Input · [{inputShape.join(", ")}]
      </div>

      {small ? (
        <>
          <div className="grid grid-cols-2 gap-2">
            {vals.map((v, i) => (
              <label key={i} className="flex items-center gap-1 text-[11px] text-neutral-500">
                x{i}
                <input
                  className={field}
                  value={v}
                  onChange={(e) => setVals((a) => a.map((x, j) => (j === i ? Number(e.target.value) : x)))}
                />
              </label>
            ))}
          </div>
          <div className="flex gap-2">
            <button className={btn + " bg-violet-500 text-neutral-950 hover:bg-violet-400"} disabled={busy} onClick={() => run(modelId, { kind: "vector", values: vals })}>
              Run
            </button>
            <button className={btn + " border border-neutral-700 text-neutral-300 hover:bg-neutral-800"} onClick={() => setVals((a) => a.map(() => Number((Math.random() * 2 - 1).toFixed(2))))}>
              Randomize
            </button>
            <button className={btn + " border border-neutral-700 text-neutral-300 hover:bg-neutral-800"} onClick={() => setVals((a) => a.map(() => 0))}>
              Zeros
            </button>
          </div>
        </>
      ) : (
        <>
          {isImage && (
            <>
              {(archs.length ? archs : ["alexnet"]).map((arch) => (
                <button
                  key={arch}
                  className={btn + " w-full border border-amber-500/50 text-amber-200 hover:bg-amber-500/10"}
                  disabled={busy}
                  onClick={() => onPretrained(arch)}
                >
                  Load pretrained {arch.toUpperCase()} weights
                </button>
              ))}
              {archs.length > 1 && (
                <div className="text-center text-[10px] text-neutral-600">pick the one matching your drawn network (load its example first)</div>
              )}
              {pre && <div className="rounded-lg border border-neutral-700 bg-neutral-800/50 p-2 text-[11px] text-neutral-300">{pre}</div>}
              <input ref={imgRef} type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && onImage(e.target.files[0])} />
              <button
                className={btn + " w-full bg-violet-500 text-neutral-950 hover:bg-violet-400"}
                disabled={busy}
                onClick={() => imgRef.current?.click()}
              >
                Upload an image to classify
              </button>
              <div className="text-center text-[10px] text-neutral-600">resized to {inputShape.join("×")} automatically · or sample a dataset below</div>
            </>
          )}
          <div
            onClick={() => fileRef.current?.click()}
            className="cursor-pointer rounded-xl border-2 border-dashed border-neutral-700 p-3 text-center text-xs text-neutral-400 hover:border-neutral-600"
          >
            <input ref={fileRef} type="file" accept=".csv,.json,.npy,.npz,.zip" className="hidden" onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
            {datasetId ? `dataset loaded · ${nSamples} samples` : isImage ? "or an image ZIP (class folders) to sample" : "Upload a dataset to sample an input"}
          </div>
          {datasetId && (
            <>
              <label className="block text-[11px] text-neutral-500">
                sample #{index}
                <input type="range" min={0} max={Math.max(0, nSamples - 1)} value={index} onChange={(e) => setIndex(Number(e.target.value))} className="mt-1 w-full accent-violet-500" />
              </label>
              <button className={btn + " w-full bg-violet-500 text-neutral-950 hover:bg-violet-400"} disabled={busy} onClick={() => run(modelId, { kind: "dataset", dataset_id: datasetId, index })}>
                Run sample #{index}
              </button>
            </>
          )}
          <button className={btn + " w-full border border-neutral-700 text-neutral-300 hover:bg-neutral-800"} disabled={busy} onClick={() => run(modelId, { kind: "random", seed: Math.floor(Math.random() * 1e6) })}>
            Run random input
          </button>
        </>
      )}
      {err && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-2 text-[11px] text-red-300">{err}</div>}
    </div>
  );
}
