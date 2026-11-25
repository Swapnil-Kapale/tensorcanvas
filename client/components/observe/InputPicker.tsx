"use client";

import { useEffect, useRef, useState } from "react";
import { useObserve } from "@/lib/observeStore";
import { uploadDataset } from "@/lib/trainApi";
import { listPretrained, loadPretrained } from "@/lib/observeApi";

const parseVec = (t: string): number[] =>
  t.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean).map(Number).filter((n) => !Number.isNaN(n));

export function InputPicker({
  modelId,
  inputShape,
  hasConv,
}: {
  modelId: string;
  inputShape: number[];
  hasConv: boolean;
}) {
  const run = useObserve((s) => s.run);
  const busy = useObserve((s) => s.busy);
  const dim = inputShape.reduce((a, b) => a * b, 1);

  // Input UI adapts to the network: a conv network (CNN) takes an image; a
  // dense network (ANN) takes numeric values. Pretrained weights only apply to
  // ImageNet-shaped conv nets (3×224×224) — don't offer them otherwise.
  const showImage = hasConv;
  const showNumeric = !hasConv;
  const isImageNet = inputShape.length === 3 && inputShape[0] === 3 && inputShape[1] === 224 && inputShape[2] === 224;
  const bigVector = showNumeric && dim > 64;

  const [vals, setVals] = useState<number[]>([]);
  const [text, setText] = useState("");
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [nSamples, setNSamples] = useState(0);
  const [index, setIndex] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [pre, setPre] = useState<string | null>(null);
  const [archs, setArchs] = useState<string[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLInputElement>(null);

  // Pretrained archs only matter for ImageNet-shaped conv nets.
  useEffect(() => {
    if (!isImageNet) return;
    let alive = true;
    listPretrained().then((a) => { if (alive && a.length) setArchs(a); }).catch(() => {});
    return () => { alive = false; };
  }, [isImageNet]);

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

  const runVector = () => {
    const values = bigVector ? parseVec(text) : Array.from({ length: dim }, (_, i) => vals[i] ?? 0);
    if (values.length !== dim) {
      setErr(`this network needs ${dim} input value${dim === 1 ? "" : "s"}, got ${values.length}`);
      return;
    }
    setErr(null);
    run(modelId, { kind: "vector", values });
  };

  const field = "w-full rounded-md border border-[var(--stroke)] bg-black/25 px-2 py-1 text-right font-mono text-xs text-[var(--text)] outline-none focus:border-violet-500";
  const btn = "rounded-lg px-3 py-1.5 text-sm font-semibold transition disabled:opacity-40";
  const outline = btn + " border border-[var(--stroke)] text-[var(--muted)] hover:bg-[var(--glass-hover)]";
  const primary = btn + " tc-btn-primary";

  return (
    <div className="space-y-3 border-b border-[var(--stroke)] p-4">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--faint)]">
          Input · [{inputShape.join(", ")}]
        </span>
        <span className="rounded bg-black/30 px-1.5 py-0.5 text-[10px] text-[var(--muted)]">
          {showImage ? "image network" : "numeric network"}
        </span>
      </div>

      {/* ── Numeric entry for dense / ANN networks ── */}
      {showNumeric && (
        <>
          {bigVector ? (
            <>
              <textarea
                className="h-28 w-full rounded-md border border-[var(--stroke)] bg-black/25 p-2 font-mono text-xs text-[var(--text)] outline-none focus:border-violet-500"
                placeholder={`Enter ${dim} numbers (comma- or space-separated)`}
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <div className="text-[10px] text-[var(--faint)]">{parseVec(text).length}/{dim} values</div>
            </>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {Array.from({ length: dim }, (_, i) => vals[i] ?? 0).map((v, i) => (
                <label key={i} className="flex items-center gap-1 text-[11px] text-[var(--faint)]">
                  x{i}
                  <input
                    className={field}
                    value={v}
                    onChange={(e) =>
                      setVals((a) => {
                        const b = Array.from({ length: dim }, (_, j) => a[j] ?? 0);
                        b[i] = Number(e.target.value);
                        return b;
                      })
                    }
                  />
                </label>
              ))}
            </div>
          )}
          <div className="flex gap-2">
            <button className={primary} disabled={busy} onClick={runVector}>Run</button>
            <button
              className={outline}
              onClick={() => {
                const r = () => Number((Math.random() * 2 - 1).toFixed(2));
                if (bigVector) setText(Array.from({ length: dim }, r).join(", "));
                else setVals(Array.from({ length: dim }, r));
              }}
            >
              Randomize
            </button>
            <button
              className={outline}
              onClick={() => {
                if (bigVector) setText(Array.from({ length: dim }, () => 0).join(", "));
                else setVals(Array.from({ length: dim }, () => 0));
              }}
            >
              Zeros
            </button>
          </div>
        </>
      )}

      {/* ── Image entry for conv / CNN networks ── */}
      {showImage && (
        <>
          {isImageNet &&
            (archs.length ? archs : ["alexnet"]).map((arch) => (
              <button
                key={arch}
                className={btn + " w-full border border-amber-500/50 text-amber-200 hover:bg-amber-500/10"}
                disabled={busy}
                onClick={() => onPretrained(arch)}
              >
                Load pretrained {arch.toUpperCase()} weights
              </button>
            ))}
          {isImageNet && archs.length > 1 && (
            <div className="text-center text-[10px] text-[var(--faint)]">pick the one matching your drawn network (load its example first)</div>
          )}
          {pre && <div className="rounded-lg border border-[var(--stroke)] bg-black/25 p-2 text-[11px] text-[var(--muted)]">{pre}</div>}
          <input ref={imgRef} type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && onImage(e.target.files[0])} />
          <button className={primary + " w-full"} disabled={busy} onClick={() => imgRef.current?.click()}>
            Upload an image to classify
          </button>
          <div className="text-center text-[10px] text-[var(--faint)]">resized to {inputShape.join("×")} automatically · or sample a dataset below</div>
        </>
      )}

      {/* ── Dataset sampling + random — available for every network ── */}
      <div
        onClick={() => fileRef.current?.click()}
        className="cursor-pointer rounded-xl border-2 border-dashed border-[var(--stroke)] p-3 text-center text-xs text-[var(--muted)] hover:border-[var(--stroke-strong)]"
      >
        <input ref={fileRef} type="file" accept=".csv,.json,.npy,.npz,.zip" className="hidden" onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])} />
        {datasetId ? `dataset loaded · ${nSamples} samples` : showImage ? "or an image ZIP (class folders) to sample" : "or upload a dataset to sample an input"}
      </div>
      {datasetId && (
        <>
          <label className="block text-[11px] text-[var(--faint)]">
            sample #{index}
            <input type="range" min={0} max={Math.max(0, nSamples - 1)} value={index} onChange={(e) => setIndex(Number(e.target.value))} className="mt-1 w-full accent-violet-500" />
          </label>
          <button className={primary + " w-full"} disabled={busy} onClick={() => run(modelId, { kind: "dataset", dataset_id: datasetId, index })}>
            Run sample #{index}
          </button>
        </>
      )}
      <button className={outline + " w-full"} disabled={busy} onClick={() => run(modelId, { kind: "random", seed: Math.floor(Math.random() * 1e6) })}>
        Run random input
      </button>

      {err && <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-2 text-[11px] text-red-300">{err}</div>}
    </div>
  );
}
