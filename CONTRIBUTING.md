# Contributing to TensorCanvas

Thanks for your interest in improving TensorCanvas! It is built to be extended —
adding a new layer, dataset format, pretrained model, or example should be a
small, well-scoped change. This guide shows you where things live and how to get
a change merged.

By participating you agree to our [Code of Conduct](./CODE_OF_CONDUCT.md).

## Development setup

You need **uv**, **Node 20+** with Corepack (`corepack enable`), and **Git**.

```bash
# clone your fork, then:
./dev.sh          # installs everything on first run, starts both servers
```

Or set each side up by hand:

```bash
cd server && uv sync                 # backend deps into server/.venv
cd client && yarn install            # frontend deps
```

- Frontend → http://localhost:3000
- Backend  → http://localhost:8000 (API docs at `/docs`)

## Before you open a PR

```bash
cd server && uv run pytest           # backend tests must pass
cd client && yarn lint && yarn build # frontend must lint & build clean
```

- Backend tests live in `server/tests/`. Please add a test for new behaviour.
  Tests must **not** download pretrained weights (the ~233 MB download is
  exercised manually only).
- The frontend has no test runner yet — adding one is a welcome **good first
  issue**.
- A Python linter (`ruff`) is not yet wired up; keep style consistent with
  surrounding code. Adding `ruff` + CI is also a welcome contribution.

## Where to contribute

TensorCanvas has a few well-defined extension points. Each has a recipe in
[`docs/`](./docs/):

| I want to add… | Touch these files | Recipe |
|---|---|---|
| A **layer type** | `server/.../builder.py` (`LAYERS`) **and** `client/components/build/nodeCatalog.ts` (`CATALOG`, `PALETTE_ITEMS`) | [adding-a-layer.md](./docs/adding-a-layer.md) |
| A **dataset format** | `server/.../datasets.py` (a loader + the dispatch) | [adding-a-dataset.md](./docs/adding-a-dataset.md) |
| A **pretrained model** | `server/.../pretrained.py` (`PRETRAINED` registry) + a matching example | [adding-a-pretrained-model.md](./docs/adding-a-pretrained-model.md) |
| An **example network** | `client/lib/examples.ts` | [adding-an-example.md](./docs/adding-an-example.md) |
| A **workspace/tab** | `client/components/WorkspaceShell.tsx` + a `components/<area>/` folder + a store in `client/lib/` | see [ARCHITECTURE.md](./ARCHITECTURE.md) |

> **Tip:** to try a backend-only layer *without* touching the frontend, use the
> **Custom node** in the palette — it sends any backend `type` you name.

Read **[ARCHITECTURE.md](./ARCHITECTURE.md)** first — it describes the coupling
points (especially the layer-catalog two-file edit) so your change doesn't break
silently.

## Pull request flow

1. **Fork** and create a branch: `git checkout -b feat/my-change`.
2. Make your change; add/adjust tests; update docs if you touched an extension
   point.
3. Run tests, lint, and build (see above).
4. Open a PR using the template. Describe **what** and **why**, and include
   screenshots/GIFs for UI changes.
5. If you added a layer, confirm you updated **both** `builder.py:LAYERS` and
   `nodeCatalog.ts` (the PR checklist reminds you).

## Commit messages

We use short, PR-style, conventional-ish subjects:

```
feat(server): add BatchNorm2d layer
fix(client): delete edge on Backspace
docs: document adding a dataset format
```

Signing off (`git commit -s`) is appreciated but not required. A one-line SPDX
header on new source files (`# SPDX-License-Identifier: Apache-2.0`) is
encouraged.

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache License 2.0](./LICENSE).
