"""JSON graph  →  validated PyTorch ``nn.Module``.

The build happens in three passes so that failures are reported with the most
useful message possible:

1. **Structural validation** — unknown layer types, duplicate / malformed ids,
   edges pointing at missing nodes, exactly one input and one output, a single
   linear chain (v1), and no cycles. All structural problems are collected and
   returned together.
2. **Shape inference** — walk the chain from the input shape, computing each
   node's output shape and validating compatibility.
3. **Construction** — instantiate the real ``torch.nn`` modules with the
   inferred dimensions (applying bias / activation / initial weights) and wrap
   them in a :class:`GraphModule`.

``in_features`` / ``in_channels`` are never supplied by the client; they are
inferred in pass 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn

from .schema import NetworkSpec

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

Shape = tuple[int, ...]  # a tensor shape WITHOUT the leading batch dimension


@dataclass
class BuildProblem:
    code: str
    message: str
    node_id: str | None = None


class BuildFailed(Exception):
    def __init__(self, problems: list[BuildProblem]):
        self.problems = problems
        super().__init__(f"{len(problems)} problem(s) building network")


# ── small param helpers ──────────────────────────────────────────────────────

def _as_int(params: dict, key: str, node_id: str, *, required=True, default=None) -> int:
    if key not in params:
        if required:
            raise ValueError(f"missing required parameter '{key}'")
        return default
    val = params[key]
    if not isinstance(val, int) or isinstance(val, bool):
        raise ValueError(f"parameter '{key}' must be an integer, got {val!r}")
    return val


def _as_bool(params: dict, key: str, *, default: bool = True) -> bool:
    if key not in params or params[key] is None:
        return default
    val = params[key]
    if not isinstance(val, bool):
        raise ValueError(f"parameter '{key}' must be a boolean, got {val!r}")
    return val


def _conv_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


# ── activations (usable as a layer's `activation` attribute) ─────────────────

ACTIVATIONS: dict[str, object] = {
    "none": None,
    "linear": None,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": lambda: nn.Softmax(dim=-1),
}


def _activation_by_name(name: object) -> nn.Module | None:
    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError(f"activation must be a string, got {name!r}")
    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {name!r}; choose from {', '.join(sorted(set(ACTIVATIONS)))}"
        )
    factory = ACTIVATIONS[key]
    return None if factory is None else factory()  # type: ignore[operator]


def _make_activation(params: dict) -> nn.Module | None:
    return _activation_by_name(params.get("activation", "none"))


class PerNeuronActivation(nn.Module):
    """Applies a (possibly different) activation to each output neuron."""

    def __init__(self, acts: list[nn.Module]):
        super().__init__()
        self.acts = nn.ModuleList(acts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [..., out_features]
        cols = [self.acts[i](x[..., i : i + 1]) for i in range(len(self.acts))]
        return torch.cat(cols, dim=-1)


def _make_per_neuron_activation(na: object, out_features: int) -> nn.Module | None:
    if not isinstance(na, list):
        raise ValueError("'neuron_activations' must be a list")
    if len(na) != out_features:
        raise ValueError(
            f"'neuron_activations' has length {len(na)} but the layer has "
            f"{out_features} neurons"
        )
    mods: list[nn.Module] = []
    any_act = False
    for item in na:
        m = _activation_by_name(item)
        mods.append(m if m is not None else nn.Identity())
        if m is not None:
            any_act = True
    return PerNeuronActivation(mods) if any_act else None


def _wrap_activation(layer: nn.Module, params: dict) -> nn.Module:
    act = _make_activation(params)
    return layer if act is None else nn.Sequential(layer, act)


def core_layer(module: nn.Module) -> nn.Module:
    """The primary layer inside a built module (unwraps the activation)."""
    return module[0] if isinstance(module, nn.Sequential) else module


# ── initial weights/bias for Linear ──────────────────────────────────────────

def _to_tensor_2d(v: object, name: str) -> torch.Tensor:
    if not isinstance(v, list) or not v or not all(isinstance(r, list) for r in v):
        raise ValueError(f"'{name}' must be a 2-D array (a list of rows)")
    width = len(v[0])
    for r in v:
        if len(r) != width:
            raise ValueError(f"'{name}' rows must all have the same length")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in r):
            raise ValueError(f"'{name}' must contain only numbers")
    return torch.tensor(v, dtype=torch.float32)


def _to_tensor_1d(v: object, name: str) -> torch.Tensor:
    if not isinstance(v, list) or not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in v
    ):
        raise ValueError(f"'{name}' must be a 1-D array of numbers")
    return torch.tensor(v, dtype=torch.float32)


def _apply_linear_init(layer: nn.Linear, params: dict) -> None:
    if params.get("init_weights") is not None:
        t = _to_tensor_2d(params["init_weights"], "init_weights")
        exp = (layer.out_features, layer.in_features)
        if tuple(t.shape) != exp:
            raise ValueError(
                f"'init_weights' shape {tuple(t.shape)} does not match expected "
                f"{exp} (out_features × in_features)"
            )
        with torch.no_grad():
            layer.weight.copy_(t)
    if params.get("init_bias") is not None:
        if layer.bias is None:
            raise ValueError("'init_bias' was given but bias is disabled for this layer")
        t = _to_tensor_1d(params["init_bias"], "init_bias")
        exp = (layer.out_features,)
        if tuple(t.shape) != exp:
            raise ValueError(
                f"'init_bias' shape {tuple(t.shape)} does not match expected {exp}"
            )
        with torch.no_grad():
            layer.bias.copy_(t)


# ── Layer catalogue ──────────────────────────────────────────────────────────
# infer(in_shape, params, node_id) -> out_shape ; build(in_shape, out_shape, params) -> nn.Module
# Inference / build raise ValueError with a human message; the caller attaches the node id.

@dataclass
class LayerDef:
    infer: object
    build: object


def _infer_input(in_shape, params, node_id):
    shape = params.get("shape")
    if not isinstance(shape, list) or not shape or not all(
        isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in shape
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
            f"kernel_size={kernel} (stride={stride}, padding={padding}) is too large "
            f"for input {in_shape}: resulting size would be {out_h}x{out_w}."
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
        raise ValueError(f"kernel_size={kernel} is too large for input {in_shape}.")
    return (c, out_h, out_w)


def _infer_flatten(in_shape, params, node_id):
    prod = 1
    for d in in_shape:
        prod *= d
    return (prod,)


def _identity_shape(in_shape, params, node_id):
    return in_shape


def _build_linear(in_shape, out_shape, params):
    bias = _as_bool(params, "bias", default=True)
    layer = nn.Linear(in_shape[0], out_shape[0], bias=bias)
    _apply_linear_init(layer, params)
    na = params.get("neuron_activations")
    if na is not None:
        act = _make_per_neuron_activation(na, out_shape[0])
        return nn.Sequential(layer, act) if act is not None else layer
    return _wrap_activation(layer, params)


def _build_conv2d(in_shape, out_shape, params):
    bias = _as_bool(params, "bias", default=True)
    layer = nn.Conv2d(
        in_shape[0], params["out_channels"], params["kernel_size"],
        stride=params.get("stride", 1), padding=params.get("padding", 0), bias=bias,
    )
    return _wrap_activation(layer, params)


LAYERS: dict[str, LayerDef] = {
    "input": LayerDef(_infer_input, lambda i, o, p: nn.Identity()),
    "output": LayerDef(_identity_shape, lambda i, o, p: _wrap_activation(nn.Identity(), p)),
    "linear": LayerDef(_infer_linear, _build_linear),
    "conv2d": LayerDef(_infer_conv2d, _build_conv2d),
    "maxpool2d": LayerDef(
        _infer_maxpool2d,
        lambda i, o, p: nn.MaxPool2d(p["kernel_size"], stride=p.get("stride", p["kernel_size"])),
    ),
    "flatten": LayerDef(_infer_flatten, lambda i, o, p: nn.Flatten(start_dim=1)),
    "relu": LayerDef(_identity_shape, lambda i, o, p: nn.ReLU()),
    "sigmoid": LayerDef(_identity_shape, lambda i, o, p: nn.Sigmoid()),
    "tanh": LayerDef(_identity_shape, lambda i, o, p: nn.Tanh()),
    "softmax": LayerDef(_identity_shape, lambda i, o, p: nn.Softmax(dim=-1)),
}


# ── The executable model ─────────────────────────────────────────────────────

class GraphModule(nn.Module):
    """Runs the built layers in topological order, following the chain."""

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
    core: dict[str, nn.Module]  # node id -> primary layer (activation unwrapped)


# ── validation (unchanged structure checks) ──────────────────────────────────

def _validate_structure(spec: NetworkSpec) -> list[BuildProblem]:
    problems: list[BuildProblem] = []
    ids = [n.id for n in spec.nodes]

    if not spec.nodes:
        return [BuildProblem("empty", "The network has no nodes.")]

    seen: set[str] = set()
    for n in spec.nodes:
        if not _ID_RE.match(n.id):
            problems.append(BuildProblem(
                "invalid_id",
                f"Node id {n.id!r} is invalid; use letters, digits, '_' or '-'.",
                n.id,
            ))
        if n.id in seen:
            problems.append(BuildProblem("duplicate_id", f"Duplicate node id {n.id!r}.", n.id))
        seen.add(n.id)
        if n.type not in LAYERS:
            problems.append(BuildProblem(
                "unknown_type",
                f"Unknown layer type {n.type!r}. Supported: {', '.join(sorted(LAYERS))}.",
                n.id,
            ))

    id_set = set(ids)
    for e in spec.edges:
        if e.source not in id_set:
            problems.append(BuildProblem("bad_edge", f"Edge source {e.source!r} is not a node.", e.source))
        if e.target not in id_set:
            problems.append(BuildProblem("bad_edge", f"Edge target {e.target!r} is not a node.", e.target))

    inputs = [n.id for n in spec.nodes if n.type == "input"]
    outputs = [n.id for n in spec.nodes if n.type == "output"]
    if len(inputs) != 1:
        problems.append(BuildProblem(
            "input_count", f"Exactly one 'input' node is required; found {len(inputs)}."))
    if len(outputs) != 1:
        problems.append(BuildProblem(
            "output_count", f"Exactly one 'output' node is required; found {len(outputs)}."))

    if problems:
        return problems

    indeg = {i: 0 for i in ids}
    outdeg = {i: 0 for i in ids}
    for e in spec.edges:
        outdeg[e.source] += 1
        indeg[e.target] += 1

    input_id, output_id = inputs[0], outputs[0]
    for n in spec.nodes:
        exp_in = 0 if n.id == input_id else 1
        exp_out = 0 if n.id == output_id else 1
        if indeg[n.id] != exp_in:
            problems.append(BuildProblem(
                "bad_degree",
                f"Node {n.id!r} should have {exp_in} incoming connection(s), has "
                f"{indeg[n.id]}. (v1 supports a single linear chain.)",
                n.id,
            ))
        if outdeg[n.id] != exp_out:
            problems.append(BuildProblem(
                "bad_degree",
                f"Node {n.id!r} should have {exp_out} outgoing connection(s), has "
                f"{outdeg[n.id]}. (v1 supports a single linear chain.)",
                n.id,
            ))
    if problems:
        return problems

    order: list[str] = []
    cur: str | None = input_id
    visited: set[str] = set()
    succ = {e.source: e.target for e in spec.edges}
    while cur is not None:
        if cur in visited:
            return [BuildProblem("cycle", "The network contains a cycle.")]
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
    problems = _validate_structure(spec)
    if problems:
        raise BuildFailed(problems)

    nodes = {n.id: n for n in spec.nodes}
    input_id = next(n.id for n in spec.nodes if n.type == "input")
    output_id = next(n.id for n in spec.nodes if n.type == "output")
    succ = {e.source: e.target for e in spec.edges}

    order: list[str] = []
    pred: dict[str, str | None] = {input_id: None}
    cur: str | None = input_id
    prev: str | None = None
    while cur is not None:
        order.append(cur)
        pred[cur] = prev
        prev, cur = cur, succ.get(cur)

    layers: dict[str, nn.Module] = {}
    core: dict[str, nn.Module] = {}
    output_shapes: dict[str, Shape] = {}
    in_shape: Shape = ()
    for nid in order:
        node = nodes[nid]
        layerdef = LAYERS[node.type]
        try:
            out_shape = layerdef.infer(in_shape, node.params, nid)
        except ValueError as exc:
            raise BuildFailed([BuildProblem("shape_mismatch", str(exc), nid)]) from exc
        try:
            module = layerdef.build(in_shape, out_shape, node.params)
        except ValueError as exc:
            raise BuildFailed([BuildProblem("shape_mismatch", str(exc), nid)]) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise BuildFailed([BuildProblem("build_error", str(exc), nid)]) from exc
        layers[nid] = module
        core[nid] = core_layer(module)
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
        core=core,
    )

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
