"use client";

import { useEffect, useRef } from "react";
import { CATEGORY_ORDER, PALETTE_ITEMS, type NodeCategory, type PaletteItem } from "./nodeCatalog";

export interface PaletteMenuProps {
  x: number;
  y: number;
  onPick: (item: PaletteItem) => void;
  onClose: () => void;
}

export function PaletteMenu({ x, y, onPick, onClose }: PaletteMenuProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  return (
    <div
      ref={ref}
      style={{ left: x, top: y }}
      className="fixed z-50 w-52 overflow-hidden rounded-md border border-neutral-700 bg-neutral-900 py-1 text-sm text-neutral-200 shadow-2xl"
    >
      <div className="px-3 py-1 text-[10px] uppercase tracking-wide text-neutral-500">Add node</div>
      {CATEGORY_ORDER.map((cat: NodeCategory) => {
        const items = PALETTE_ITEMS.filter((e) => e.category === cat);
        if (items.length === 0) return null;
        return (
          <div key={cat} className="py-0.5">
            <div className="px-3 pt-1 text-[10px] font-semibold uppercase tracking-wide text-neutral-600">
              {cat}
            </div>
            {items.map((e) => (
              <button
                key={e.label}
                onClick={() => onPick(e)}
                className="flex w-full items-center justify-between px-3 py-1.5 text-left hover:bg-neutral-700/70"
              >
                <span>{e.label}</span>
                <span className="font-mono text-[10px] text-neutral-500">{e.short}</span>
              </button>
            ))}
          </div>
        );
      })}
    </div>
  );
}
