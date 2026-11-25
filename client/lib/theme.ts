"use client";

import { create } from "zustand";

/** Available accent palettes. The values live in CSS (`[data-palette="…"]` in
 *  globals.css); here we only track the chosen id + a swatch for the picker. */
export const PALETTES: { id: string; label: string; hi: string; lo: string }[] = [
  { id: "platinum", label: "Platinum", hi: "#f4f7fc", lo: "#c6cedb" },
  { id: "violet", label: "Violet Surge", hi: "#9b8cff", lo: "#6d28d9" },
  { id: "blue", label: "Electric Blue", hi: "#4f8cff", lo: "#2563eb" },
  { id: "teal", label: "Mint Teal", hi: "#4fe3c8", lo: "#0d9488" },
  { id: "ember", label: "Ember", hi: "#fdba74", lo: "#ea580c" },
  { id: "fuchsia", label: "Fuchsia Neon", hi: "#f0abfc", lo: "#c026d3" },
];

export const DEFAULT_PALETTE = "platinum";
const STORAGE_KEY = "tensorcanvas:palette";

function initial(): string {
  if (typeof window === "undefined") return DEFAULT_PALETTE;
  try {
    return window.localStorage.getItem(STORAGE_KEY) || DEFAULT_PALETTE;
  } catch {
    return DEFAULT_PALETTE;
  }
}

interface ThemeState {
  palette: string;
  setPalette: (id: string) => void;
}

export const useTheme = create<ThemeState>((set) => ({
  palette: initial(),
  setPalette: (id) => {
    try {
      window.localStorage.setItem(STORAGE_KEY, id);
    } catch {
      /* ignore */
    }
    if (typeof document !== "undefined") document.documentElement.dataset.palette = id;
    set({ palette: id });
  },
}));
