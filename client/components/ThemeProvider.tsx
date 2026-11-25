"use client";

import { useEffect } from "react";
import { useTheme } from "@/lib/theme";

/** Applies the selected palette to <html data-palette>. A blocking inline script
 *  in layout.tsx sets it before paint (no flash); this keeps it in sync on change. */
export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const palette = useTheme((s) => s.palette);
  useEffect(() => {
    document.documentElement.dataset.palette = palette;
  }, [palette]);
  return <>{children}</>;
}
