# Manual testing & running guide

How to run TensorCanvas and manually verify every feature end-to-end. For the
automated suite, see the bottom.

## 1. Prerequisites

- **Node.js 20+** with Corepack: `corepack enable` (provides Yarn 4)
- **uv** — https://docs.astral.sh/uv/ (fetches Python 3.12 automatically)
- **Git**

No accounts, keys, or secrets are needed.

## 2. Run the app

```bash
./dev.sh
```

First run installs both sides (the Python side pulls PyTorch/torchvision CPU
wheels — hundreds of MB, slow once). Then:

- Frontend → http://localhost:3000
- Backend  → http://localhost:8000  (interactive API docs at http://localhost:8000/docs)

`Ctrl+C` stops both. To use different ports: `FRONTEND_PORT=4000 BACKEND_PORT=9000 ./dev.sh`
(the client is pointed at the backend automatically).

### Backend-only sanity checks (optional)

```bash
curl -s localhost:8000/health                 # {"status":"ok",...}
curl -s localhost:8000/api/v1/catalog          # layer types + activations the backend builds
curl -s localhost:8000/api/v1/pretrained       # {"archs":["alexnet","vgg16"]}
```

## 3. Manual test flows (UI)

Open http://localhost:3000. Work left-to-right through the tabs: **Build → Train →
Observe**.

### A. Build & compile
1. **Build** tab → **Load example → ANN → MLP**. A graph appears
   (Input → Flatten → Dense → Dense → Output).
2. Click a **Dense** node → confirm you can see/edit neurons, weights, bias, and
   the activation. Use the toolbar to add/delete a node or edge, and undo/redo.
3. Click **Compile**. Expect success. ✔
4. **Negative test:** delete an edge so the graph is disconnected, Compile again →
   the offending node is flagged red with a specific error message. Undo to fix.

### B. Train (ANN, tabular data)
1. With the MLP compiled, go to **Train**.
2. **Upload dataset** → `sample-data/example-shapes.csv`
   (regenerate first if missing: `cd sample-data && uv run --project ../server python generate.py`).
3. Set epochs / optimizer / loss, **Start**. Expect: live loss/accuracy updating
   over SSE, and neurons glowing by activation. Pause / resume / stop should work;
   accuracy should climb (3 easily separable classes).

### C. Train (CNN, image data)
1. **Build → Load example → CNN → Small CNN (trainable)** → **Compile**.
2. **Train → Upload** `sample-data/shapes-images.zip` (class-folder images) → **Start**.
   Expect training to converge similarly.

### D. Observe / run / intervene
1. With a trained model, open **Observe**.
2. Pick an input (type a vector, sample the dataset, or **Run random input**) and
   press **Play**. Expect the forward pass to animate layer-by-layer: neurons light
   violet as they "compute," activation values are labelled, and the **Output**
   panel shows class probabilities + the prediction.
3. Select a Dense layer → inspect its activations (large layers show the **top-64
   by magnitude** with a "show all" toggle).
4. **Snapshots:** each run is saved in the Snapshot browser; load an older one.
5. **Live edit:** edit a weight (live), re-run, and confirm the prediction changes.

### E. Pretrained image classification (the headline demo)
1. **Build → Load example → CNN → AlexNet** (or **VGG-16**) → **Compile**
   (large param count — real architecture).
2. **Observe → Load pretrained AlexNet weights** (or VGG-16). First time downloads
   the weights (~233 MB AlexNet / ~528 MB VGG-16) — **pre-warm before a demo**.
3. **Upload an image** (any real photo). Expect the forward pass to animate and the
   **top-5 ImageNet predictions with human-readable labels** to appear. Inputs are
   ImageNet-normalized automatically, so predictions should be sensible.
4. **Negative test:** load the AlexNet example but click **Load pretrained VGG-16** →
   a clear "architecture doesn't match" error (the drawn graph must match the arch).

### F. Import / export architecture
1. **Build → Export** → saves a `.tcnet.json` (architecture only, no weights).
2. Edit/clear the canvas, then **Import** that file → the graph is restored.
3. **Negative test:** try importing `sample-data/example-shapes.csv` as a network,
   or uploading `alexnet.tcnet.json` as a *dataset* → each gives a clear, specific
   error pointing you to the right place.

## 4. Automated tests

```bash
cd server && uv run pytest        # backend suite (offline; never downloads weights)
cd client && yarn lint            # frontend lint
cd client && yarn build           # production build must succeed
```

All three must be clean before any change is considered done.

## 5. Troubleshooting

- **Port already in use:** set `FRONTEND_PORT` / `BACKEND_PORT` (see §2).
- **`yarn: command not found`:** run `corepack enable`.
- **First `uv sync` is very slow:** that's the one-time Torch/torchvision download.
- **Client can't reach the backend:** check `NEXT_PUBLIC_API_URL` (defaults to
  `http://localhost:8000`); `dev.sh` sets it from `BACKEND_PORT`.
- **Pretrained load hangs the first time:** it's downloading weights; pre-warm it
  before a live demo. Cache location is `~/.cache/torch` (override with `TORCH_HOME`).
- **Models/snapshots vanished:** state is in-memory; restarting the backend clears
  all built models, runs, and snapshots. Rebuild/retrain to continue.
