"use client";

import { useObserve } from "@/lib/observeStore";

export function ForwardPassPlayer() {
  const { snapshot, stepIndex, playing, speed, loop, play, pause, step, seek, setSpeed, setLoop } = useObserve();
  if (!snapshot) return null;
  const n = snapshot.order.length;
  const current = snapshot.order[Math.max(0, stepIndex)] ?? "—";

  const btn = "h-8 rounded-lg border border-[var(--stroke)] px-2.5 text-sm text-[var(--muted)] hover:bg-[var(--glass-hover)] disabled:opacity-30";

  return (
    <div className="tc-glass flex items-center gap-2 rounded-2xl px-4 py-2">
      <button className={btn} onClick={() => seek(0)} title="restart">⏮</button>
      <button className={btn} onClick={() => step(-1)} disabled={stepIndex <= 0} title="step back (←)">◀</button>
      {playing ? (
        <button className={btn + " !border-[var(--violet)]/50 !text-[var(--violet)]"} onClick={pause} title="pause (space)">⏸</button>
      ) : (
        <button className={btn + " !border-[var(--violet)]/50 !text-[var(--violet)]"} onClick={play} title="play (space)">▶</button>
      )}
      <button className={btn} onClick={() => step(1)} disabled={stepIndex >= n - 1} title="step forward (→)">▶▶</button>

      {/* timeline */}
      <input
        type="range"
        min={0}
        max={n - 1}
        value={Math.max(0, stepIndex)}
        onChange={(e) => seek(Number(e.target.value))}
        className="mx-2 flex-1 accent-violet-500"
      />
      <span className="w-28 shrink-0 font-mono text-[11px] text-[var(--muted)]">
        {Math.max(0, stepIndex) + 1}/{n} · {current}
      </span>

      <label className="flex items-center gap-1 text-[11px] text-[var(--muted)]">
        <input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} className="h-3.5 w-3.5 accent-violet-500" />
        loop
      </label>
      <select
        value={speed}
        onChange={(e) => setSpeed(Number(e.target.value))}
        className="h-8 rounded-lg border border-[var(--stroke)] bg-black/30 px-1.5 text-xs text-[var(--text)]"
        title="speed"
      >
        {[0.5, 1, 2, 4].map((x) => (
          <option key={x} value={x}>{x}×</option>
        ))}
      </select>
    </div>
  );
}
