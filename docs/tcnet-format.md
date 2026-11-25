# The `.tcnet.json` network file format

TensorCanvas exports and imports network **architectures** as `.tcnet.json`.
A file is just the canvas graph — nodes and edges — and maps 1:1 to the backend
build endpoint. The definitions live in `client/lib/examples.ts`
(`graphToFile` / `fileToGraph`).

> **Architecture only — not weights.** A `.tcnet.json` describes the *shape* of a
> network, not its trained parameters. Share it to share a design; it is small and
> human-readable.

## Shape

```jsonc
{
  "format": "tensorcanvas-net",   // identifies the file; must be exactly this
  "version": 1,                   // NET_VERSION
  "name": "my-net",
  "nodes": [
    {
      "id": "in",
      "type": "input",             // a backend layer type (LAYERS key), or "custom"
      "position": { "x": 0, "y": 0 },
      "params": { "shape": "1, 28, 28" },  // string values (as edited in the UI)
      "activation": "relu",        // optional
      "bias": true,                // optional
      "customType": ""             // only for type "custom": the real backend type
    }
    // ...
  ],
  "edges": [
    { "source": "in", "target": "flat" }
    // ...
  ]
}
```

| Field | Notes |
|---|---|
| `format` | Must be `"tensorcanvas-net"`. Import rejects anything else with a clear error. |
| `version` | Currently `1`. |
| `name` | Display name for the network. |
| `nodes[].type` | A backend layer type (see [adding-a-layer.md](./adding-a-layer.md)) or `"custom"`. |
| `nodes[].position` | Canvas coordinates, so the layout round-trips. |
| `nodes[].params` | Per-layer parameters as **strings** (the UI edits them as text). |
| `nodes[].customType` | Only for `type: "custom"` — the actual backend type to build. |
| `edges` | Logical connections; the backend topologically orders them. |

## Gotcha: this is not a dataset

A `.tcnet.json` is an architecture, **not** training data. If you upload one to a
dataset field, the backend detects it and returns a clear message pointing you to
the Build canvas's import instead. (Datasets are CSV/JSON/NPY/NPZ/image-ZIP — see
[adding-a-dataset.md](./adding-a-dataset.md).)

## Programmatic use

- **Export:** `graphToFile(nodes, edges, name)` → a `NetFile` object you can
  `JSON.stringify`.
- **Import:** `fileToGraph(parsedJson)` → a `Graph` (`{ nodes, edges }`) ready for
  the canvas, validating `format`/`version` along the way.
