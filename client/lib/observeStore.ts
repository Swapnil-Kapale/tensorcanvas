"use client";

import { create } from "zustand";
import { useGraph } from "@/lib/store";
import {
  deleteSnapshot as apiDelete,
  getSnapshot,
  listSnapshots,
  runModel,
  type ObserveSnapshot,
  type RunInput,
  type SnapshotMeta,
} from "@/lib/observeApi";

let ticker: ReturnType<typeof setTimeout> | null = null;

// Write the forward-pass state into the shared graph store for nodes up to `k`.
function applyFrame(snap: ObserveSnapshot, k: number) {
  const g = useGraph.getState();
  const pos: Record<string, number> = {};
  snap.order.forEach((id, i) => (pos[id] = i));
  useGraph.setState({
    nodes: g.nodes.map((n) => {
      const i = pos[n.id];
      if (i === undefined) return n;
      if (i > k) {
        return { ...n, data: { ...n.data, activationMag: null, observePhase: "idle", observeValues: null } };
      }
      const rec = snap.nodes[n.id];
      return {
        ...n,
        data: {
          ...n.data,
          activationMag: rec?.mag ?? null,
          observePhase: i === k ? "computing" : "done",
          observeValues: rec?.values ?? null,
        },
      };
    }),
  });
}

function clearGraphObserve() {
  const g = useGraph.getState();
  useGraph.setState({
    nodes: g.nodes.map((n) => ({ ...n, data: { ...n.data, activationMag: null, observePhase: null, observeValues: null } })),
  });
}

interface ObserveState {
  snapshot: ObserveSnapshot | null;
  snapshots: SnapshotMeta[];
  stepIndex: number; // -1 idle, else index into snapshot.order
  playing: boolean;
  speed: number;
  loop: boolean;
  lastInput: RunInput | null;
  weightVersion: number;
  busy: boolean;
  error: string | null;
  selectedNode: string | null;

  run: (modelId: string, input: RunInput, opts?: { label?: string }) => Promise<void>;
  loadSnapshot: (id: string) => Promise<void>;
  refreshList: (modelId: string) => Promise<void>;
  remove: (id: string, modelId: string) => Promise<void>;
  play: () => void;
  pause: () => void;
  step: (dir: number) => void;
  seek: (i: number) => void;
  setSpeed: (x: number) => void;
  setLoop: (b: boolean) => void;
  selectNode: (id: string | null) => void;
  clear: () => void;
}

export const useObserve = create<ObserveState>((set, get) => ({
  snapshot: null,
  snapshots: [],
  stepIndex: -1,
  playing: false,
  speed: 1,
  loop: false,
  lastInput: null,
  weightVersion: 0,
  busy: false,
  error: null,
  selectedNode: null,

  run: async (modelId, input, opts) => {
    set({ busy: true, error: null, lastInput: input });
    try {
      const snap = await runModel(modelId, input, { label: opts?.label });
      set({ snapshot: snap, stepIndex: -1, weightVersion: snap.weight_version, busy: false, selectedNode: snap.order[snap.order.length - 1] ?? null });
      get().refreshList(modelId);
      get().play();
    } catch (e) {
      set({ busy: false, error: e instanceof Error ? e.message : String(e) });
    }
  },

  loadSnapshot: async (id) => {
    set({ busy: true, error: null });
    try {
      const snap = await getSnapshot(id);
      set({ snapshot: snap, stepIndex: -1, busy: false, selectedNode: snap.order[snap.order.length - 1] ?? null });
      get().play();
    } catch (e) {
      set({ busy: false, error: e instanceof Error ? e.message : String(e) });
    }
  },

  refreshList: async (modelId) => {
    try {
      set({ snapshots: await listSnapshots(modelId) });
    } catch {
      /* ignore */
    }
  },

  remove: async (id, modelId) => {
    await apiDelete(id).catch(() => {});
    get().refreshList(modelId);
  },

  play: () => {
    const s = get();
    if (!s.snapshot) return;
    if (ticker) clearTimeout(ticker);
    set({ playing: true });
    if (s.stepIndex >= s.snapshot.order.length - 1) {
      applyFrame(s.snapshot, 0);
      set({ stepIndex: 0 });
    }
    const tick = () => {
      const st = get();
      if (!st.playing || !st.snapshot) return;
      const next = st.stepIndex + 1;
      if (next >= st.snapshot.order.length) {
        if (st.loop) {
          applyFrame(st.snapshot, 0);
          set({ stepIndex: 0 });
        } else {
          set({ playing: false });
          return;
        }
      } else {
        applyFrame(st.snapshot, next);
        set({ stepIndex: next });
      }
      ticker = setTimeout(tick, 650 / Math.max(0.25, get().speed));
    };
    ticker = setTimeout(tick, 650 / Math.max(0.25, s.speed));
  },

  pause: () => {
    if (ticker) clearTimeout(ticker);
    set({ playing: false });
  },

  step: (dir) => {
    const s = get();
    if (!s.snapshot) return;
    get().pause();
    get().seek(s.stepIndex + dir);
  },

  seek: (i) => {
    const s = get();
    if (!s.snapshot) return;
    const k = Math.max(0, Math.min(s.snapshot.order.length - 1, i));
    applyFrame(s.snapshot, k);
    set({ stepIndex: k, selectedNode: s.snapshot.order[k] });
  },

  setSpeed: (x) => set({ speed: x }),
  setLoop: (b) => set({ loop: b }),
  selectNode: (id) => set({ selectedNode: id }),

  clear: () => {
    if (ticker) clearTimeout(ticker);
    ticker = null;
    clearGraphObserve();
    set({ snapshot: null, stepIndex: -1, playing: false, selectedNode: null, error: null });
  },
}));
