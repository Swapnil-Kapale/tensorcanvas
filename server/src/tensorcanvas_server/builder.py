"""JSON graph  →  validated PyTorch ``nn.Module``.

The build happens in three passes so that failures are reported with the most
useful message possible:

1. **Structural validation** — unknown layer types, duplicate / malformed ids,
   edges pointing at missing nodes, exactly one input and one output, a single
   linear chain (v1), and no cycles. All structural problems are collected and
   returned together.
2. **Shape inference** — walk the chain from the input shape, computing each
   node's output shape and validating compatibility (this is where "Linear
   expects 1-D input" style errors come from). Fails at the first bad node,
   since everything downstream depends on it.
3. **Construction** — instantiate the real ``torch.nn`` modules with the
   inferred dimensions and wrap them in a :class:`GraphModule`.

``in_features`` / ``in_channels`` are never supplied by the client; they are
inferred in pass 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import nn

from .schema import NetworkSpec

# Node ids become ``nn.ModuleDict`` keys, which may not contain "." and must be
# non-empty. Keep them to a sane, canvas-friendly character set.
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

Shape = tuple[int, ...]  # a tensor shape WITHOUT the leading batch dimension


@dataclass
class BuildProblem:
    """A single validation error (mirrors schema.BuildError)."""

    code: str
    message: str
    node_id: str | None = None


class BuildFailed(Exception):
    """Raised when a network cannot be built. Carries structured problems."""

    def __init__(self, problems: list[BuildProblem]):
        self.problems = problems
        super().__init__(f"{len(problems)} problem(s) building network")


# ── Layer catalogue ────────────────────────────────────────────────────────
# Each entry knows how to (a) infer its output shape from its input shape and
# params, and (b) construct the corresponding torch module. Inference functions
# raise ValueError with a human message; the caller attaches the node id.


def _as_int(params: dict, key: str, node_id: str, *, required=True, default=None) -> int:
    if key not in params:
        if required:
            raise ValueError(f"missing required parameter '{key}'")
        return default
    val = params[key]
    if not isinstance(val, int) or isinstance(val, bool):
        raise ValueError(f"parameter '{key}' must be an integer, got {val!r}")
    return val


def _conv_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


@dataclass
class LayerDef:
    infer: callable  # (in_shape, params, node_id) -> out_shape
    build: callable  # (in_shape, params) -> nn.Module


def _infer_input(in_shape, params, node_id):
    shape = params.get("shape")
    if not isinstance(shape, list) or not shape or not all(
        isinstance(d, int) and d > 0 for d in shape
    ):
        raise ValueError(
            "input node requires a 'shape' list of positive integers "
            "(the per-sample shape, excluding batch), e.g. [1, 28, 28]"
        )
    return tuple(shape)


def _infer_linear(in_shape, params, node_id):
    out_features = _as_int(params, "out_features", node_id)
    if out_features <= 0:
        raise ValueError("'out_features' must be positive")
    if len(in_shape) != 1:
        raise ValueError(
            f"Linear expects a 1-D input [features]; got shape {in_shape}. "
            "Insert a 'flatten' before this layer."
        )
    return (out_features,)


def _infer_conv2d(in_shape, params, node_id):
    out_channels = _as_int(params, "out_channels", node_id)
    kernel = _as_int(params, "kernel_size", node_id)
    stride = _as_int(params, "stride", node_id, required=False, default=1)
    padding = _as_int(params, "padding", node_id, required=False, default=0)
    if len(in_shape) != 3:
        raise ValueError(
            f"Conv2d expects a 3-D input [channels, height, width]; got {in_shape}."
        )
    _, h, w = in_shape
    out_h = _conv_out(h, kernel, stride, padding)
    out_w = _conv_out(w, kernel, stride, padding)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"kernel_size={kernel} (stride={stride}, padding={padding}) is too "
            f"large for input {in_shape}: resulting size would be "
            f"{out_h}x{out_w}."
        )
    return (out_channels, out_h, out_w)


def _infer_maxpool2d(in_shape, params, node_id):
    kernel = _as_int(params, "kernel_size", node_id)
    stride = _as_int(params, "stride", node_id, required=False, default=kernel)
    if len(in_shape) != 3:
        raise ValueError(
            f"MaxPool2d expects a 3-D input [channels, height, width]; got {in_shape}."
        )
    c, h, w = in_shape
    out_h = _conv_out(h, kernel, stride, 0)
    out_w = _conv_out(w, kernel, stride, 0)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"kernel_size={kernel} is too large for input {in_shape}."
        )
    return (c, out_h, out_w)


def _infer_flatten(in_shape, params, node_id):
    prod = 1
    for d in in_shape:
        prod *= d
    return (prod,)


def _identity_shape(in_shape, params, node_id):
    return in_shape


LAYERS: dict[str, LayerDef] = {
    "input": LayerDef(_infer_input, lambda s, p: nn.Identity()),
    "output": LayerDef(_identity_shape, lambda s, p: nn.Identity()),
    "linear": LayerDef(
        _infer_linear, lambda s, p: nn.Linear(s[0], p["out_features"])
    ),
    "conv2d": LayerDef(
        _infer_conv2d,
        lambda s, p: nn.Conv2d(
            s[0], p["out_channels"], p["kernel_size"],
            stride=p.get("stride", 1), padding=p.get("padding", 0),
        ),
    ),
    "maxpool2d": LayerDef(
        _infer_maxpool2d,
        lambda s, p: nn.MaxPool2d(p["kernel_size"], stride=p.get("stride", p["kernel_size"])),
    ),
    "flatten": LayerDef(_infer_flatten, lambda s, p: nn.Flatten(start_dim=1)),
    "relu": LayerDef(_identity_shape, lambda s, p: nn.ReLU()),
    "sigmoid": LayerDef(_identity_shape, lambda s, p: nn.Sigmoid()),
    "tanh": LayerDef(_identity_shape, lambda s, p: nn.Tanh()),
    "softmax": LayerDef(_identity_shape, lambda s, p: nn.Softmax(dim=-1)),
}


# ── The executable model ─────────────────────────────────────────────────────

class GraphModule(nn.Module):
    """Runs the built layers in topological order, following the chain.

    Intermediate outputs are kept in a dict during ``forward`` keyed by node id
    — the hook point future "Observe"/"Intervene" features will build on.
    """

    def __init__(
        self,
        layers: dict[str, nn.Module],
        order: list[str],
        pred: dict[str, str | None],
        input_id: str,
        output_id: str,
    ):
        super().__init__()
        self.layers = nn.ModuleDict(layers)
        self.order = order
        self.pred = pred
        self.input_id = input_id
        self.output_id = output_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations: dict[str, torch.Tensor] = {}
        for nid in self.order:
            src = x if self.pred[nid] is None else activations[self.pred[nid]]
            activations[nid] = self.layers[nid](src)
        return activations[self.output_id]


@dataclass
class BuiltModel:
    module: GraphModule
    name: str | None
    input_shape: Shape
    order: list[str]
    node_types: dict[str, str]
    output_shapes: dict[str, Shape]


# ── The three passes ─────────────────────────────────────────────────────────

def _validate_structure(spec: NetworkSpec) -> list[BuildProblem]:
    problems: list[BuildProblem] = []
    ids = [n.id for n in spec.nodes]

    if not spec.nodes:
        return [BuildProblem("empty", "The network has no nodes.")]

    # id sanity + duplicates
    seen: set[str] = set()
    for n in spec.nodes:
        if not _ID_RE.match(n.id):
            problems.append(BuildProblem(
                "invalid_id",
                f"Node id {n.id!r} is invalid; use letters, digits, '_' or '-'.",
                n.id,
            ))
        if n.id in seen:
            problems.append(BuildProblem(
                "duplicate_id", f"Duplicate node id {n.id!r}.", n.id))
        seen.add(n.id)
        if n.type not in LAYERS:
            problems.append(BuildProblem(
                "unknown_type",
                f"Unknown layer type {n.type!r}. Supported: "
                f"{', '.join(sorted(LAYERS))}.",
                n.id,
            ))

    id_set = set(ids)
    for e in spec.edges:
        if e.source not in id_set:
            problems.append(BuildProblem(
                "bad_edge", f"Edge source {e.source!r} is not a node.", e.source))
        if e.target not in id_set:
            problems.append(BuildProblem(
                "bad_edge", f"Edge target {e.target!r} is not a node.", e.target))

    inputs = [n.id for n in spec.nodes if n.type == "input"]
    outputs = [n.id for n in spec.nodes if n.type == "output"]
    if len(inputs) != 1:
        problems.append(BuildProblem(
            "input_count",
            f"Exactly one 'input' node is required; found {len(inputs)}."))
    if len(outputs) != 1:
        problems.append(BuildProblem(
            "output_count",
            f"Exactly one 'output' node is required; found {len(outputs)}."))

    # Stop here if anything above is wrong — degree/cycle checks assume a clean
    # node/edge set.
    if problems:
        return problems

    indeg: dict[str, int] = {i: 0 for i in ids}
    outdeg: dict[str, int] = {i: 0 for i in ids}
    pred: dict[str, list[str]] = {i: [] for i in ids}
    for e in spec.edges:
        outdeg[e.source] += 1
        indeg[e.target] += 1
        pred[e.target].append(e.source)

    input_id, output_id = inputs[0], outputs[0]
    for n in spec.nodes:
        exp_in = 0 if n.id == input_id else 1
        exp_out = 0 if n.id == output_id else 1
        if indeg[n.id] != exp_in:
            problems.append(BuildProblem(
                "bad_degree",
                f"Node {n.id!r} should have {exp_in} incoming connection(s), "
                f"has {indeg[n.id]}. (v1 supports a single linear chain.)",
                n.id,
            ))
        if outdeg[n.id] != exp_out:
            problems.append(BuildProblem(
                "bad_degree",
                f"Node {n.id!r} should have {exp_out} outgoing connection(s), "
                f"has {outdeg[n.id]}. (v1 supports a single linear chain.)",
                n.id,
            ))
    if problems:
        return problems

    # Cycle / reachability: walk the unique chain from the input.
    order: list[str] = []
    cur: str | None = input_id
    visited: set[str] = set()
    succ = {e.source: e.target for e in spec.edges}
    while cur is not None:
        if cur in visited:
            problems.append(BuildProblem(
                "cycle", "The network contains a cycle."))
            return problems
        visited.add(cur)
        order.append(cur)
        cur = succ.get(cur)
    if len(order) != len(ids):
        unreached = sorted(set(ids) - visited)
        problems.append(BuildProblem(
            "disconnected",
            "These nodes are not connected into the input→output chain: "
            f"{', '.join(unreached)}.",
        ))
    return problems


def build_network(spec: NetworkSpec) -> BuiltModel:
    """Validate and build. Raises :class:`BuildFailed` with structured problems."""

    problems = _validate_structure(spec)
    if problems:
        raise BuildFailed(problems)

    nodes = {n.id: n for n in spec.nodes}
    input_id = next(n.id for n in spec.nodes if n.type == "input")
    output_id = next(n.id for n in spec.nodes if n.type == "output")
    succ = {e.source: e.target for e in spec.edges}

    # Chain order from input.
    order: list[str] = []
    pred: dict[str, str | None] = {input_id: None}
    cur: str | None = input_id
    prev: str | None = None
    while cur is not None:
        order.append(cur)
        pred[cur] = prev
        prev, cur = cur, succ.get(cur)

    # Pass 2 + 3: infer shapes and construct modules along the chain.
    layers: dict[str, nn.Module] = {}
    output_shapes: dict[str, Shape] = {}
    in_shape: Shape = ()
    for nid in order:
        node = nodes[nid]
        layer = LAYERS[node.type]
        try:
            out_shape = layer.infer(in_shape, node.params, nid)
        except ValueError as exc:
            raise BuildFailed([BuildProblem("shape_mismatch", str(exc), nid)]) from exc
        try:
            module = layer.build(out_shape if node.type == "input" else in_shape, node.params)
        except Exception as exc:  # pragma: no cover - defensive
            raise BuildFailed([BuildProblem("build_error", str(exc), nid)]) from exc
        layers[nid] = module
        output_shapes[nid] = out_shape
        in_shape = out_shape

    module = GraphModule(layers, order, pred, input_id, output_id)

    built = BuiltModel(
        module=module,
        name=spec.name,
        input_shape=output_shapes[input_id],
        order=order,
        node_types={n.id: n.type for n in spec.nodes},
        output_shapes=output_shapes,
    )

    # Final safety net: a real forward pass with a dummy batch. Catches anything
    # shape inference missed and guarantees the model actually runs.
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, *built.input_shape)
            module.eval()(dummy)
    except Exception as exc:
        raise BuildFailed([
            BuildProblem("runtime_check", f"Network failed a test forward pass: {exc}")
        ]) from exc

    return built


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
