"use client";

import { useState } from "react";
import { BuildCanvas } from "./build/BuildCanvas";

const WORKSPACES = ["Build", "Observe", "Explore", "Intervene"] as const;
type Workspace = (typeof WORKSPACES)[number];

export function WorkspaceShell() {
  const [active, setActive] = useState<Workspace>("Build");

  return (
    <div className="flex h-full min-h-0 flex-col bg-neutral-950 text-neutral-100">
      <header className="flex items-center gap-4 border-b border-neutral-800 px-4">
        <span className="py-3 text-sm font-semibold tracking-tight">
          Tensor<span className="text-sky-400">Canvas</span>
        </span>
        <nav className="flex gap-1">
          {WORKSPACES.map((w) => {
            const enabled = w === "Build";
            const isActive = w === active;
            return (
              <button
                key={w}
                disabled={!enabled}
                onClick={() => enabled && setActive(w)}
                className={[
                  "relative px-3 py-3 text-sm transition-colors",
                  isActive ? "text-white" : "text-neutral-400",
                  enabled ? "hover:text-white" : "cursor-not-allowed opacity-40",
                ].join(" ")}
                title={enabled ? undefined : "Coming soon"}
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
        ) : (
          <div className="flex h-full items-center justify-center text-neutral-600">
            {active} workspace — coming soon
          </div>
        )}
      </main>
    </div>
  );
}
