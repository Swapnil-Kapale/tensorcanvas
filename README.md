# TensorCanvas

### Open-source neural network exploration & intervention studio

**Explore. Understand. Intervene.**

TensorCanvas makes the internal computation of neural networks understandable and
experimentally accessible. Build networks visually, run inference, inspect
activations and gradients, and intervene on weights and intermediate values to see
how the model's behaviour changes — all in the browser.

> Status: early development.

---

## Repository layout

```
tensorcanvas/
├── client/    # Next.js + TypeScript + Tailwind frontend (the studio UI)
├── server/    # Python backend (uv-managed) — model execution & inspection
└── dev.sh     # one command to run the whole app
```

## Prerequisites

- **Node.js** 20+ with **Corepack** (`corepack enable` → provides Yarn 4)
- **uv** (Python package/runtime manager) — https://docs.astral.sh/uv/
- **Python** 3.12 (uv will fetch it automatically)

## Quickstart

```bash
./dev.sh
```

That's it. On first run it installs dependencies for both sides, then starts:

- **Frontend** → http://localhost:3000
- **Backend**  → http://localhost:8000

Press **Ctrl+C** to stop both.

## Development

| | Frontend (`client/`) | Backend (`server/`) |
|---|---|---|
| Install | `yarn install` | `uv sync` |
| Run | `yarn dev` | `uv run tensorcanvas-server` |
| Build | `yarn build` | — |

Dependencies are added incrementally, as features require them.

## Contributors

- **Swapnil Kapale** ([@Swapnil-Kapale](https://github.com/Swapnil-Kapale))
- **Aryan Tandon** ([@AryTan003](https://github.com/AryTan003))
