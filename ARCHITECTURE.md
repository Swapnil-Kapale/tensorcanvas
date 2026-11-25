# Architecture

This document explains how TensorCanvas fits together, the data model it is built
around, and — importantly for contributors — where the **coupling points** are.

A useful one-line framing: **TensorCanvas is a graph compiler with a drawing UI
bolted onto each side of a thin JSON wire format.** You draw a graph in the
browser; the backend compiles that graph into a PyTorch model and runs/trains/
inspects it; results stream back.

```
  ┌─────────────────────────────┐        HTTP + SSE         ┌──────────────────────────────┐
  │  Browser — Next.js client    │ ───────────────────────▶ │  FastAPI backend              │
  │  Build / Train / Observe UI  │                          │  build · train · observe      │
  │  (React Flow canvas, Zustand)│ ◀─────────────────────── │  ── compiles graph ──▶ PyTorch │
  └─────────────────────────────┘      JSON / event stream  └──────────────────────────────┘
```

## The wire format: `.tcnet.json`

Everything flows through one JSON shape — a network **spec**:

```jsonc
{
  "name": "my-net",
  "nodes": [
    { "id": "in",   "type": "input",  "params": { "shape": [1, 28, 28] } },
    { "id": "flat", "type": "flatten" },
    { "id": "fc1",  "type": "linear", "params": { "out_features": 128 },
      "activation": "relu", "bias": true },
    { "id": "out",  "type": "output", "activation": "softmax" }
  ],
  "edges": [
    { "source": "in",   "target": "flat" },
    { "source": "flat", "target": "fc1" },
    { "source": "fc1",  "target": "out" }
  ]
}
```

- **nodes** carry a `type` (must be a backend layer type), free-form `params`, and
  optional `activation` / `bias`.
- **edges** are logical connections; the backend topologically orders them.
- The same shape is saved/loaded by the UI's import/export (`.tcnet.json`,
  `format: "tensorcanvas-net"`). See [`docs/tcnet-format.md`](./docs/tcnet-format.md).

## Backend (`server/src/tensorcanvas_server/`)

| Module | Responsibility |
|---|---|
| `schema.py` | Pydantic models for the wire format (`NetworkSpec`, `NodeSpec`, …) and build responses. |
| `builder.py` | The compiler. `LAYERS` registry (type → infer-shape + build-module), activation registry, `build_network()`, and `GraphModule` (the runnable `nn.Module` with `forward_capture()` that returns every node's activations). |
| `registry.py` | In-memory `ModelRegistry` of built models (`ModelRecord` holds the module, a `train_lock`, `weight_version`, and optional `labels`). |
| `datasets.py` | Dataset **loaders** (CSV/JSON/NPY/NPZ/image-ZIP), built-in generators, and `match_to_network()` (coerce a dataset to the model's input/output). |
| `train_schema.py` | Pydantic `TrainConfig` (optimizer/loss/scheduler/epochs/…). |
| `training.py` | The training loop, metrics, and helpers (activations, terminal activation). |
| `runs.py` | `RunRegistry` + Server-Sent-Events streaming of live training. |
| `observe.py` | `run_model()` (one forward pass → bounded, visualizable snapshot), prediction logic, and the `SnapshotRegistry`. |
| `pretrained.py` | `PRETRAINED` registry of torchvision loaders (currently AlexNet) + the generic `copy_conv_linear_in_order()` helper. |
| `main.py` | The FastAPI app and all HTTP endpoints. |
| `__init__.py` | `main()` console-script entry (`uv run tensorcanvas-server` → uvicorn). |

### Key endpoints (`/api/v1`)

- `POST /models/build` — validate + build a spec → `model_id` (or `422` with
  per-node problems).
- `GET  /catalog` — the node types + activations the backend can build (the
  source of truth; see coupling below).
- `POST /datasets`, `POST /datasets/builtin` — upload / generate a dataset.
- `POST /models/{id}/train`, `GET /runs/{id}/stream` — start training / stream it.
- `POST /models/{id}/run` — run one input → forward-pass snapshot.
- `GET/DELETE /snapshots/...`, `PATCH /models/{id}/nodes/{nid}/weights` — snapshots
  and live weight edits.
- `POST /models/{id}/weights/pretrained` — load pretrained weights.

## Frontend (`client/`)

| Area | Files | Responsibility |
|---|---|---|
| Shell | `components/WorkspaceShell.tsx` | Tabs: Build / Train / Observe / Properties. |
| Build | `components/build/*` | `BuildCanvas`, `FlowNode`, `NeuronSphere`, `nodeCatalog.ts` (palette), `edit/LayerEditor` (weight editor). |
| Train | `components/train/*` + `lib/trainStore.ts`, `lib/trainApi.ts` | Upload, configure, live training over SSE. |
| Observe | `components/observe/*` + `lib/observeStore.ts`, `lib/observeApi.ts` | Forward-pass player, inspectors, output panel, snapshot browser. |
| State / API | `lib/store.ts` (`useGraph`), `lib/api.ts`, `lib/examples.ts` | Graph state, API base URL + wire types, example nets + `.tcnet.json` I/O. |

### Key flows

1. **Build → Compile:** `useGraph` → `nodeCatalog` default data → `api.ts` builds a
   `NetworkSpec` → `POST /models/build` → node errors painted on the canvas.
2. **Train (SSE):** `trainApi` starts a run → `trainStore` subscribes to
   `/runs/{id}/stream` → per-epoch metrics + per-neuron glow pushed into `useGraph`.
3. **Observe / intervene:** `observeApi.runModel` → one snapshot → `observeStore`
   replays it frame by frame into `useGraph`; `patchNodeWeights` edits live, then
   re-runs.
4. **Pretrained:** draw the matching example → `loadPretrained` copies torchvision
   weights in place → upload an image → top-k labelled prediction.

## Known coupling points

TensorCanvas is deliberately small, but a few places must change together. These
are documented honestly so contributions don't break silently.

1. **Layer catalog (the big one).** A new drawable layer type lives in **two**
   places: the backend `builder.py:LAYERS` (how to infer shape + build the module)
   and the frontend `components/build/nodeCatalog.ts` (`CATALOG` + `PALETTE_ITEMS`
   — how it appears/edits in the palette). They can drift.
   - **Guard:** `GET /api/v1/catalog` exposes the backend types as the source of
     truth, and `server/tests/test_catalog.py` fails if the palette references a
     type the backend can't build.
   - **Escape hatch:** the **Custom node** sends an arbitrary backend `type`
     without any frontend change — use it to try a backend-only layer immediately.
   - Recipe: [`docs/adding-a-layer.md`](./docs/adding-a-layer.md).
2. **Activations** are listed in `builder.py:ACTIVATIONS` and mirrored in
   `nodeCatalog.ts` (`ACTIVATIONS` + `ACTIVATION_GLYPH`). Keep them in sync.
3. **Loss / optimizer / scheduler** are declared in `train_schema.py` (Pydantic
   `Literal`s) and dispatched in `training.py`. A mismatch fails loudly with a
   `422`, so this is low-risk.
4. **Wire types** in `lib/api.ts` ↔ `schema.py` and `lib/trainApi.ts` ↔
   `train_schema.py` are hand-mirrored; change them together. (Generating them from
   the backend OpenAPI schema is a good future improvement.)
5. **Datasets:** adding a new *format* is cheap (one loader + one dispatch line in
   `datasets.py`). Adding a new *kind* of data (beyond tensors/images) is more
   involved — it touches `match_to_network`, `preview`, and `observe`.

## Extending TensorCanvas

Step-by-step recipes live in [`docs/`](./docs/):

- [Add a layer type](./docs/adding-a-layer.md)
- [Add a dataset format](./docs/adding-a-dataset.md)
- [Add a pretrained model](./docs/adding-a-pretrained-model.md)
- [Add an example network](./docs/adding-an-example.md)
- [The `.tcnet.json` format](./docs/tcnet-format.md)
