"use client";

import { useEffect, useRef } from "react";
import type { Boundary } from "@/lib/trainStore";

const CLASS_COLORS = [
  [56, 189, 248],
  [167, 139, 250],
  [52, 211, 153],
  [251, 191, 36],
  [244, 114, 182],
  [248, 113, 113],
];

export function DecisionBoundary({
  boundary,
  scatter,
  size = 380,
}: {
  boundary: Boundary | null;
  scatter?: { points: number[][]; labels: number[] | null } | null;
  size?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    c.width = size;
    c.height = size;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, size, size);

    const bounds = boundary?.bounds ?? boundsOf(scatter);
    if (!bounds) {
      ctx.fillStyle = "rgba(255,255,255,0.25)";
      ctx.font = "12px ui-sans-serif";
      ctx.fillText("Train a 2-D classifier to see the decision boundary", 16, 24);
      return;
    }

    if (boundary) {
      const { res, values, mode } = boundary;
      const img = ctx.createImageData(res, res);
      for (let i = 0; i < res * res; i++) {
        let r: number, g: number, b: number;
        if (mode === "prob") {
          const p = Math.max(0, Math.min(1, values[i]));
          const a = CLASS_COLORS[0];
          const z = CLASS_COLORS[1];
          r = a[0] + (z[0] - a[0]) * p;
          g = a[1] + (z[1] - a[1]) * p;
          b = a[2] + (z[2] - a[2]) * p;
        } else {
          [r, g, b] = CLASS_COLORS[values[i] % CLASS_COLORS.length];
        }
        img.data[i * 4] = r;
        img.data[i * 4 + 1] = g;
        img.data[i * 4 + 2] = b;
        img.data[i * 4 + 3] = 100;
      }
      const off = document.createElement("canvas");
      off.width = res;
      off.height = res;
      off.getContext("2d")!.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.save();
      ctx.translate(0, size);
      ctx.scale(1, -1); // ymin at bottom
      ctx.drawImage(off, 0, 0, size, size);
      ctx.restore();
    }

    if (scatter?.points?.length) {
      const { xmin, xmax, ymin, ymax } = bounds;
      const sx = (x: number) => ((x - xmin) / (xmax - xmin || 1)) * size;
      const sy = (y: number) => size - ((y - ymin) / (ymax - ymin || 1)) * size;
      scatter.points.forEach((p, idx) => {
        const lbl = scatter.labels ? scatter.labels[idx] : 0;
        const col = CLASS_COLORS[lbl % CLASS_COLORS.length];
        ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
        ctx.beginPath();
        ctx.arc(sx(p[0]), sy(p[1]), 2.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,0.55)";
        ctx.lineWidth = 0.6;
        ctx.stroke();
      });
    }
  }, [boundary, scatter, size]);

  return (
    <canvas ref={ref} className="rounded-xl border border-[var(--stroke)]" style={{ width: size, height: size }} />
  );
}

function boundsOf(scatter?: { points: number[][] } | null) {
  if (!scatter?.points?.length) return null;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const [x, y] of scatter.points) {
    xmin = Math.min(xmin, x);
    xmax = Math.max(xmax, x);
    ymin = Math.min(ymin, y);
    ymax = Math.max(ymax, y);
  }
  const mx = (xmax - xmin) * 0.1 + 1e-3;
  const my = (ymax - ymin) * 0.1 + 1e-3;
  return { xmin: xmin - mx, xmax: xmax + mx, ymin: ymin - my, ymax: ymax + my };
}
