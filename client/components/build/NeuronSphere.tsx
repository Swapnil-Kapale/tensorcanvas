"use client";

// CSS "3D sphere" neuron — mirrors the structure of frontend-3d/NeuronSphere.tsx
// (glow halo + shaded core + inner highlight + hover/select ring). No colors copied.

export function NeuronSphere({
  size = 30,
  mag = null,
  selected = false,
  dim = false,
}: {
  size?: number;
  mag?: number | null; // 0..1 weight magnitude → brightness
  selected?: boolean;
  dim?: boolean;
}) {
  const brightness = mag == null ? 0.35 : 0.3 + mag * 0.7;
  return (
    <div className="relative" style={{ width: size, height: size }}>
      {/* glow halo */}
      <div
        className="absolute rounded-full bg-sky-400/30 blur-[6px]"
        style={{ inset: -size * 0.22, opacity: dim ? 0.2 : 0.5 }}
      />
      {/* shaded core: light top-left → dark bottom-right */}
      <div
        className="absolute inset-0 rounded-full"
        style={{
          background:
            "radial-gradient(circle at 32% 28%, " +
            `rgba(125,211,252,${brightness}) 0%, ` +
            `rgba(56,189,248,${brightness * 0.9}) 42%, ` +
            "rgba(12,74,110,0.95) 100%)",
          boxShadow: "inset -2px -3px 5px rgba(0,0,0,0.55), 0 1px 2px rgba(0,0,0,0.5)",
        }}
      />
      {/* inner highlight */}
      <div
        className="absolute rounded-full bg-white"
        style={{
          width: size * 0.22,
          height: size * 0.22,
          left: size * 0.26,
          top: size * 0.22,
          opacity: 0.75,
          filter: "blur(0.5px)",
        }}
      />
      {/* hover/select ring */}
      {selected && (
        <div className="absolute rounded-full ring-2 ring-white/80" style={{ inset: -3 }} />
      )}
    </div>
  );
}
