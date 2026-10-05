"""TensorCanvas backend — FastAPI application.

Exposes the network-building endpoint the canvas talks to:

    POST /api/v1/models/build   JSON network spec  →  built model or errors
    GET  /health                liveness probe

On a valid spec the model is built, registered, and a ``model_id`` is returned
so it can later be run/trained. On an invalid spec a 422 is returned with a
list of specific problems.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from torch import nn

from .builder import BuildFailed, build_network, count_parameters
from .registry import registry
from .schema import (
    BuildError,
    BuildFailure,
    BuildSuccess,
    NetworkSpec,
    NodeInfo,
)

# Echo full weight values only for small layers, to bound the response payload.
_WEIGHT_ECHO_LIMIT = 4096

app = FastAPI(
    title="TensorCanvas API",
    version="0.1.0",
    description="Build and inspect neural networks drawn on the canvas.",
)

# The Next.js dev frontend runs on :3000 and calls this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "models_built": registry.count()}


@app.post(
    "/api/v1/models/build",
    tags=["models"],
    response_model=BuildSuccess,
    responses={422: {"model": BuildFailure}},
    summary="Build a neural network from a JSON graph",
)
def build_model(spec: NetworkSpec):
    """Validate and construct the network.

    Returns ``200`` with the built model's summary, or ``422`` with a list of
    problems describing exactly what is wrong with the structure.
    """
    try:
        built = build_network(spec)
    except BuildFailed as exc:
        failure = BuildFailure(
            errors=[
                BuildError(code=p.code, message=p.message, node_id=p.node_id)
                for p in exc.problems
            ]
        )
        return JSONResponse(status_code=422, content=failure.model_dump())

    model_id = registry.add(built)
    nodes = [_node_info(built, nid) for nid in built.order]
    return BuildSuccess(
        model_id=model_id,
        name=built.name,
        num_parameters=count_parameters(built.module),
        nodes=nodes,
    )


@app.get("/api/v1/models/{model_id}/nodes/{node_id}/weights", tags=["models"])
def node_weights(model_id: str, node_id: str):
    """Full weight matrix + bias for one node — used by the weight-edit page for
    layers too large to echo in the build response."""
    rec = registry.get(model_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"detail": "model not found"})
    built = rec.built
    core = built.core.get(node_id)
    if core is None:
        return JSONResponse(status_code=404, content={"detail": "node not found"})
    if not isinstance(core, nn.Linear):
        return JSONResponse(status_code=400, content={"detail": "node has no weight matrix"})
    return {
        "node_id": node_id,
        "weight_shape": [core.out_features, core.in_features],
        "has_bias": core.bias is not None,
        "weights": core.weight.detach().tolist(),
        "bias_values": core.bias.detach().tolist() if core.bias is not None else None,
    }


def _node_info(built, nid: str) -> NodeInfo:
    core = built.core[nid]
    info = NodeInfo(
        id=nid,
        type=built.node_types[nid],
        output_shape=list(built.output_shapes[nid]),
        num_parameters=count_parameters(built.module.layers[nid]),
    )
    if isinstance(core, nn.Linear):
        info.has_bias = core.bias is not None
        info.weight_shape = [core.out_features, core.in_features]
        if core.out_features * core.in_features <= _WEIGHT_ECHO_LIMIT:
            info.weights = core.weight.detach().tolist()
            if core.bias is not None:
                info.bias_values = core.bias.detach().tolist()
    elif isinstance(core, nn.Conv2d):
        info.has_bias = core.bias is not None
    return info
