# Recipe: add an example network

Examples populate the **Load example** menu on the Build canvas. They are
frontend-only, defined in `client/lib/examples.ts`.

## 1. Write a builder function

An example is a function returning a `Graph` (nodes + edges). Follow the existing
`annExample()` / `smallCnnExample()` / `alexnetExample()`:

```ts
export function tinyMlpExample(): Graph {
  // build nodes (id, type, params, activation/bias) and edges (source → target)
  // return { nodes, edges };
}
```

Each node's `type` must be a type the backend can build (a `LAYERS` key — see
[adding-a-layer.md](./adding-a-layer.md)), and the graph must be shape-consistent
end to end (Input → … → Output), or compiling it will report an error.

## 2. Register it in `EXAMPLES`

`EXAMPLES` is grouped (e.g. `ANN`, `CNN`). Add your entry to a group, or add a new
group:

```ts
export const EXAMPLES = [
  { group: "ANN", items: [
    { id: "mlp",  label: "MLP (28×28 → 3)", make: () => annExample() },
    { id: "tiny", label: "Tiny MLP",        make: () => tinyMlpExample() }, // ← new
  ]},
  // ...
];
```

- `id` must be unique; `label` is what users see in the menu.
- `make` is called when the user picks the example.

## 3. Verify

```bash
cd client && yarn lint && yarn build
```

Then `./dev.sh` → Build → **Load example** → pick yours → **Compile** and confirm
it builds without errors.

> If your example is the matching architecture for a pretrained model, make sure
> its conv/linear layer shapes match the reference exactly — see
> [adding-a-pretrained-model.md](./adding-a-pretrained-model.md).
