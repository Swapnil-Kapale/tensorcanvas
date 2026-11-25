# Recipe: add a new layer type

A drawable layer type lives in **two** places that must agree — the backend
(how to build it) and the frontend palette (how to draw/edit it). This is the
single most important coupling point in TensorCanvas; a drift guard test keeps
you honest.

> **Shortcut — try it with zero frontend changes first.** The **Custom node** in
> the palette sends an arbitrary backend `type` + params. If you only add the
> backend half (step 1), you can already drop a Custom node, set its type to your
> new layer, and compile. Promote it to a first-class palette entry (step 2) once
> it works.

## 1. Backend — teach the compiler to build it

Edit `server/src/tensorcanvas_server/builder.py`.

Each layer needs an **infer** function (output shape from input shape + params,
raising `ValueError` on bad params) and a **build** function (returns the
`nn.Module`). Register both in `LAYERS`.

```python
def _infer_batchnorm2d(in_shape, params, node_id):
    if len(in_shape) != 3:
        raise ValueError(f"BatchNorm2d expects [C, H, W]; got {in_shape}.")
    return in_shape  # shape-preserving

LAYERS["batchnorm2d"] = LayerDef(
    _infer_batchnorm2d,
    lambda in_shape, out_shape, params: nn.BatchNorm2d(in_shape[0]),
)
```

- For a shape-preserving layer you can reuse the existing `_identity_shape`.
- If your layer has weights you want users to inspect/edit, make sure it ends up
  in `built.core` the way `linear`/`conv2d` do (see `_build_linear`).

Add a backend test in `server/tests/` that builds a tiny graph using the new
type and asserts the output shape.

## 2. Frontend — add it to the palette

Edit `client/components/build/nodeCatalog.ts`. Add a `CatalogEntry` to `CATALOG`
and an entry to `PALETTE_ITEMS`:

```ts
// in CATALOG
{
  type: "batchnorm2d",      // MUST equal the backend LAYERS key
  label: "BatchNorm2D",
  category: "Layers",
  short: "BN",
  fields: [],               // add FieldSpec entries for any editable params
},
// in PALETTE_ITEMS
{ label: "BatchNorm2D", type: "batchnorm2d", short: "BN", category: "Layers" },
```

`fields` drive the node's editor inputs; each `FieldSpec.key` must match the
backend param name.

## 3. Verify

```bash
cd server && uv run pytest        # includes the catalog drift guard
cd client && yarn lint && yarn build
```

`server/tests/test_catalog.py` fails if the palette has a `type` the backend
can't build — so if you forget step 1, you'll know. Finally, run `./dev.sh`, drop
your layer on the canvas, wire it up, and compile.
