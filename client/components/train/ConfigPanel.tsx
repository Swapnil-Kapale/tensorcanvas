"use client";

import { useTrain } from "@/lib/trainStore";

const field =
  "w-24 rounded-md border border-neutral-700 bg-neutral-800/80 px-2 py-1 text-right font-mono text-xs text-neutral-100 outline-none focus:border-sky-500";
const sel =
  "rounded-md border border-neutral-700 bg-neutral-800/80 px-2 py-1 text-xs text-neutral-100 outline-none focus:border-sky-500";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-neutral-500">{title}</div>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex items-center justify-between gap-2 text-xs text-neutral-400">
      <span>{label}</span>
      {children}
    </label>
  );
}

export function ConfigPanel({ disabled }: { disabled: boolean }) {
  const c = useTrain((s) => s.config);
  const set = useTrain((s) => s.setConfig);
  const num = (k: keyof typeof c) => (e: React.ChangeEvent<HTMLInputElement>) => set({ [k]: Number(e.target.value) } as never);

  return (
    <div className={["space-y-5 p-4", disabled ? "pointer-events-none opacity-50" : ""].join(" ")}>
      <Section title="Training">
        <Row label="epochs"><input type="number" className={field} value={c.epochs} onChange={num("epochs")} /></Row>
        <Row label="batch size"><input type="number" className={field} value={c.batch_size} onChange={num("batch_size")} /></Row>
        <Row label="val split"><input type="number" step="0.05" className={field} value={c.val_split} onChange={num("val_split")} /></Row>
        <Row label="shuffle"><input type="checkbox" className="h-4 w-4 accent-sky-500" checked={c.shuffle} onChange={(e) => set({ shuffle: e.target.checked })} /></Row>
        <Row label="standardize X"><input type="checkbox" className="h-4 w-4 accent-sky-500" checked={c.standardize} onChange={(e) => set({ standardize: e.target.checked })} /></Row>
      </Section>

      <Section title="Loss">
        <Row label="loss">
          <select className={sel} value={c.loss} onChange={(e) => set({ loss: e.target.value })}>
            {["auto", "cross_entropy", "nll", "bce", "bce_logits", "mse", "l1", "smooth_l1"].map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </Row>
        {(c.loss === "cross_entropy" || c.loss === "auto") && (
          <Row label="label smoothing"><input type="number" step="0.05" className={field} value={c.label_smoothing} onChange={num("label_smoothing")} /></Row>
        )}
      </Section>

      <Section title="Optimizer">
        <Row label="optimizer">
          <select className={sel} value={c.optimizer} onChange={(e) => set({ optimizer: e.target.value })}>
            {["adam", "adamw", "sgd", "rmsprop"].map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </Row>
        <Row label="learning rate"><input type="number" step="0.001" className={field} value={c.lr} onChange={num("lr")} /></Row>
        <Row label="weight decay"><input type="number" step="0.0001" className={field} value={c.weight_decay} onChange={num("weight_decay")} /></Row>
        {(c.optimizer === "sgd" || c.optimizer === "rmsprop") && (
          <Row label="momentum"><input type="number" step="0.05" className={field} value={c.momentum} onChange={num("momentum")} /></Row>
        )}
        {c.optimizer === "sgd" && (
          <Row label="nesterov"><input type="checkbox" className="h-4 w-4 accent-sky-500" checked={c.nesterov} onChange={(e) => set({ nesterov: e.target.checked })} /></Row>
        )}
        {(c.optimizer === "adam" || c.optimizer === "adamw") && (
          <>
            <Row label="beta1"><input type="number" step="0.01" className={field} value={c.beta1} onChange={num("beta1")} /></Row>
            <Row label="beta2"><input type="number" step="0.001" className={field} value={c.beta2} onChange={num("beta2")} /></Row>
          </>
        )}
      </Section>

      <Section title="LR schedule">
        <Row label="scheduler">
          <select className={sel} value={c.scheduler} onChange={(e) => set({ scheduler: e.target.value })}>
            {["none", "step", "multistep", "exponential", "cosine", "cosine_warm", "plateau"].map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </Row>
        {(c.scheduler === "step" || c.scheduler === "multistep") && (
          <Row label="step size"><input type="number" className={field} value={c.step_size} onChange={num("step_size")} /></Row>
        )}
        {(c.scheduler === "step" || c.scheduler === "multistep" || c.scheduler === "exponential") && (
          <Row label="gamma"><input type="number" step="0.05" className={field} value={c.gamma} onChange={num("gamma")} /></Row>
        )}
        {(c.scheduler === "cosine" || c.scheduler === "cosine_warm") && (
          <Row label="T_max"><input type="number" className={field} value={c.t_max} onChange={num("t_max")} /></Row>
        )}
        {c.scheduler === "plateau" && (
          <>
            <Row label="patience"><input type="number" className={field} value={c.sched_patience} onChange={num("sched_patience")} /></Row>
            <Row label="factor"><input type="number" step="0.05" className={field} value={c.factor} onChange={num("factor")} /></Row>
          </>
        )}
      </Section>

      <Section title="Regularization & stability">
        <Row label="grad clip"><input type="number" step="0.5" placeholder="off" className={field} value={c.grad_clip ?? ""} onChange={(e) => set({ grad_clip: e.target.value === "" ? null : Number(e.target.value) })} /></Row>
        <Row label="grad accum"><input type="number" className={field} value={c.grad_accum} onChange={num("grad_accum")} /></Row>
        <Row label="early stopping"><input type="checkbox" className="h-4 w-4 accent-sky-500" checked={c.early_stopping.enabled} onChange={(e) => set({ early_stopping: { ...c.early_stopping, enabled: e.target.checked } })} /></Row>
        {c.early_stopping.enabled && (
          <>
            <Row label="es patience"><input type="number" className={field} value={c.early_stopping.patience} onChange={(e) => set({ early_stopping: { ...c.early_stopping, patience: Number(e.target.value) } })} /></Row>
            <Row label="restore best"><input type="checkbox" className="h-4 w-4 accent-sky-500" checked={c.early_stopping.restore_best} onChange={(e) => set({ early_stopping: { ...c.early_stopping, restore_best: e.target.checked } })} /></Row>
          </>
        )}
      </Section>

      <Section title="Init & reproducibility">
        <Row label="weight init">
          <select className={sel} value={c.init} onChange={(e) => set({ init: e.target.value })}>
            {["keep", "xavier", "kaiming", "normal"].map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </Row>
        <Row label="seed"><input type="number" className={field} value={c.seed} onChange={num("seed")} /></Row>
      </Section>
    </div>
  );
}
