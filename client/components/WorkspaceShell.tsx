"use client";

import { useState } from "react";
import { BuildCanvas } from "./build/BuildCanvas";
import { TrainWorkspace } from "./train/TrainWorkspace";
import { ObserveWorkspace } from "./observe/ObserveWorkspace";
import { useGraph } from "@/lib/store";

const WORKSPACES = ["Build", "Train", "Observe", "Explore", "Intervene", "Properties"] as const;
type Workspace = (typeof WORKSPACES)[number];

function Properties() {
  const threshold = useGraph((s) => s.renderThreshold);
  const setThreshold = useGraph((s) => s.setThreshold);
  return (
    <div className="mx-auto max-w-lg p-8">
      <h2 className="text-base font-semibold text-neutral-100">Canvas properties</h2>
      <div className="mt-6 rounded-lg border border-neutral-800 p-4">
        <label className="block">
          <span className="text-sm font-medium text-neutral-200">
            Neurons rendered individually
          </span>
          <p className="mt-1 text-xs text-neutral-500">
            Dense layers with this many neurons or fewer are drawn as individual neurons with
            per-connection edges. Larger layers collapse into a compact box.
          </p>
          <input
            type="number"
            min={1}
            max={64}
            value={threshold}
            onChange={(e) => setThreshold(Math.max(1, Number(e.target.value) || 1))}
            className="mt-3 w-28 rounded border border-neutral-700 bg-neutral-800 px-2 py-1.5 text-sm text-neutral-100 outline-none focus:border-sky-500"
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
    <div className="flex h-full min-h-0 flex-col bg-neutral-950 text-neutral-100">
      <header className="flex items-center gap-4 border-b border-neutral-800 px-4">
        <span className="py-3 text-sm font-semibold tracking-tight">
          Tensor<span className="text-sky-400">Canvas</span>
        </span>
        <nav className="flex gap-1">
          {WORKSPACES.map((w) => {
            const isActive = w === active;
            const on = enabled(w);
            return (
              <button
                key={w}
                disabled={!on}
                onClick={() => on && setActive(w)}
                className={[
                  "relative px-3 py-3 text-sm transition-colors",
                  isActive ? "text-white" : "text-neutral-400",
                  on ? "hover:text-white" : "cursor-not-allowed opacity-40",
                ].join(" ")}
                title={on ? undefined : "Coming soon"}
              >
                {w}
                {isActive && (
                  <span className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-sky-400" />
                )}
              </button>
            );
          })}
        </nav>
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
          <div className="flex h-full items-center justify-center text-neutral-600">
            {active} workspace — coming soon
          </div>
        )}
      </main>
    </div>
  );
}
