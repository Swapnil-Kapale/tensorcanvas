"use client";

// Flat, minimal neuron — a simple filled circle whose fill encodes weight/
// activation magnitude. No glow, gradient or 3D shading. `tone` picks the hue
// (sky = weights, violet = live activations during Observe).

const TONES = {
  sky: { border: "border-sky-400/70", rgb: "56,189,248" },
  violet: { border: "border-violet-400/70", rgb: "167,139,250" },
};

export function NeuronSphere({
  size = 28,
  mag = null,
  selected = false,
  tone = "sky",
}: {
  size?: number;
  mag?: number | null;
  selected?: boolean;
  tone?: "sky" | "violet";
}) {
  const opacity = mag == null ? 0.12 : 0.2 + mag * 0.75;
  const t = TONES[tone];
  return (
    <div
      className={`rounded-full border ${t.border}`}
      style={{
        width: size,
        height: size,
        background: `rgba(${t.rgb},${opacity})`,
        boxShadow: selected ? "0 0 0 2px rgba(255,255,255,0.75)" : undefined,
      }}
    />
  );
}
