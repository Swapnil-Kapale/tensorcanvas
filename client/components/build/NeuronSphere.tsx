"use client";

// Flat, minimal neuron — a simple filled circle whose fill encodes weight
// magnitude. No glow, gradient or 3D shading.

export function NeuronSphere({
  size = 28,
  mag = null,
  selected = false,
}: {
  size?: number;
  mag?: number | null;
  selected?: boolean;
}) {
  const opacity = mag == null ? 0.12 : 0.2 + mag * 0.75;
  return (
    <div
      className="rounded-full border border-sky-400/70"
      style={{
        width: size,
        height: size,
        background: `rgba(56,189,248,${opacity})`,
        boxShadow: selected ? "0 0 0 2px rgba(255,255,255,0.75)" : undefined,
      }}
    />
  );
}
