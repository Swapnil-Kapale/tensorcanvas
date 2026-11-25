"use client";

import { useState } from "react";
import { BuildCanvas } from "./build/BuildCanvas";
import { TrainWorkspace } from "./train/TrainWorkspace";
import { ObserveWorkspace } from "./observe/ObserveWorkspace";
import { useGraph } from "@/lib/store";
import { useTheme, PALETTES } from "@/lib/theme";

const WORKSPACES = ["Build", "Train", "Observe", "Explore", "Intervene", "Properties"] as const;
type Workspace = (typeof WORKSPACES)[number];

function Properties() {
  const threshold = useGraph((s) => s.renderThreshold);
  const setThreshold = useGraph((s) => s.setThreshold);
  const palette = useTheme((s) => s.palette);
  const setPalette = useTheme((s) => s.setPalette);

  return (
    <div className="mx-auto max-w-lg p-8">
      <h2 className="text-base font-semibold text-[var(--text)]">Appearance</h2>
      <div className="tc-panel mt-5 p-5">
        <span className="text-sm font-medium text-[var(--text)]">Accent palette</span>
        <p className="mt-1 text-xs text-[var(--muted)]">
          Recolors buttons, tabs and highlights. The data colors (weights, activations,
          accuracy) stay fixed so the visualizations keep their meaning.
        </p>
        <div className="mt-4 grid grid-cols-3 gap-2.5">
          {PALETTES.map((p) => {
            const active = p.id === palette;
            return (
              <button
                key={p.id}
                onClick={() => setPalette(p.id)}
                className={`flex items-center gap-2.5 rounded-xl border px-3 py-2.5 text-left text-xs transition ${
                  active
                    ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--text)]"
                    : "border-[var(--stroke)] text-[var(--muted)] hover:bg-[var(--glass-hover)]"
                }`}
              >
                <span
                  className="h-5 w-5 shrink-0 rounded-md"
                  style={{ background: `linear-gradient(160deg, ${p.hi}, ${p.lo})`, boxShadow: active ? "0 0 10px var(--accent-glow)" : undefined }}
                />
                <span className="truncate">{p.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <h2 className="mt-8 text-base font-semibold text-[var(--text)]">Canvas properties</h2>
      <div className="tc-panel mt-5 p-5">
        <label className="block">
          <span className="text-sm font-medium text-[var(--text)]">Neurons rendered individually</span>
          <p className="mt-1 text-xs text-[var(--muted)]">
            Dense layers with this many neurons or fewer are drawn as individual neurons with
            per-connection edges. Larger layers collapse into a compact box.
          </p>
          <input
            type="number"
            min={1}
            max={64}
            value={threshold}
            onChange={(e) => setThreshold(Math.max(1, Number(e.target.value) || 1))}
            className="tc-field mt-3 w-28 px-2 py-1.5 text-sm"
          />
        </label>
      </div>
    </div>
  );
}

export function WorkspaceShell() {
  const [active, setActive] = useState<Workspace>("Build");
  const enabled = (w: Workspace) =>
    w === "Build" || w === "Train" || w === "Observe" || w === "Properties";

  return (
    <div className="flex h-full min-h-0 flex-col text-[var(--text)]">
      <header className="flex items-center gap-4 border-b border-[var(--stroke)] bg-[var(--glass)] px-4 py-2 backdrop-blur-xl">
        <span className="py-1 text-sm font-semibold tracking-tight">
          Tensor<span className="text-[var(--accent)]">Canvas</span>
        </span>
        <nav className="tc-nav tc-glass mx-auto">
          {WORKSPACES.map((w) => {
            const isActive = w === active;
            const on = enabled(w);
            return (
              <button
                key={w}
                disabled={!on}
                onClick={() => on && setActive(w)}
                className={`tc-nav-item ${isActive ? "active" : ""} ${on ? "" : "cursor-not-allowed opacity-40"}`}
                title={on ? undefined : "Coming soon"}
              >
                {w}
              </button>
            );
          })}
        </nav>
        <span className="w-24" />
      </header>

      <main className="min-h-0 flex-1">
        {active === "Build" ? (
          <BuildCanvas />
        ) : active === "Train" ? (
          <TrainWorkspace />
        ) : active === "Observe" ? (
          <ObserveWorkspace />
        ) : active === "Properties" ? (
          <Properties />
        ) : (
          <div className="flex h-full items-center justify-center text-[var(--faint)]">
            {active} workspace — coming soon
          </div>
        )}
      </main>
    </div>
  );
}
