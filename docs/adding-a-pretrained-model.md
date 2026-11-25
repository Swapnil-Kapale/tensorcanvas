# Recipe: add a new pretrained model

TensorCanvas can copy real pretrained weights (from torchvision) into a graph you
drew, as long as the graph's architecture **matches** the reference model.
**AlexNet** and **VGG-16** ship today (see `pretrained.py` + their example graphs
in `client/lib/examples.ts`); this recipe shows how one was added, so you can add
another (e.g. VGG-19). Adding one is two small steps in
`server/src/tensorcanvas_server/pretrained.py`, plus a matching example so users
can get the right shape.

> **Note on ResNet:** residual/skip connections can't be expressed in the current
> linear graph format (there's no add/merge node), so ResNet isn't a drop-in.
> Pure feedforward stacks (AlexNet, VGG family) map cleanly.

## 1. Write a loader

A loader copies the reference weights into the built graph in place and returns
the list of class labels. The generic helper `copy_conv_linear_in_order()` does
the heavy lifting: it matches Conv2d/Linear layers positionally in graph order and
raises a clear `ValueError` if the drawn architecture doesn't match.

```python
def load_vgg19_into(built, record) -> list[str]:
    from torchvision.models import VGG19_Weights, vgg19

    weights = VGG19_Weights.DEFAULT
    tv = vgg19(weights=weights)
    tv.eval()
    copy_conv_linear_in_order(built, record, tv, "torchvision VGG-19")
    return list(weights.meta["categories"])
```

## 2. Register it

Add it to the `PRETRAINED` dict:

```python
PRETRAINED: dict[str, PretrainedLoader] = {
    "alexnet": load_alexnet_into,
    "vgg16": load_vgg16_into,
    "vgg19": load_vgg19_into,   # ← new
}
```

That's all the API needs — `POST /models/{id}/weights/pretrained` dispatches
through `PRETRAINED`, `GET /api/v1/pretrained` lists the available archs, the
Observe UI renders a load button per arch, and the error message lists them when
a request doesn't match.

## 3. Provide a matching example network (required)

Users can only load weights into a graph of the **exact** reference shape. Add an
example graph so they don't have to hand-draw it — see
[adding-an-example.md](./adding-an-example.md). Give it the same conv/linear
layer counts and shapes as the torchvision model (`alexnetExample()` in
`client/lib/examples.ts` is the template to copy).

## 4. The UI needs no changes

`components/observe/InputPicker.tsx` fetches `GET /api/v1/pretrained` and renders
a **Load pretrained &lt;ARCH&gt; weights** button for each registered arch
automatically. Your new entry shows up with no frontend edit — the user just
picks the one matching their drawn network.

## 5. Verify

```bash
cd server && uv run pytest
```

Keep tests **offline** — do not download weights in CI. Test the registry
wiring and the shape-mismatch `ValueError` path with a tiny fake model instead.
Then exercise the real download once, manually, through the UI.
