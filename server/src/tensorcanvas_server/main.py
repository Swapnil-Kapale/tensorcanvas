"""TensorCanvas backend — FastAPI application.

Exposes the network-building endpoint the canvas talks to:

    POST /api/v1/models/build   JSON network spec  →  built model or errors
    GET  /health                liveness probe

On a valid spec the model is built, registered, and a ``model_id`` is returned
so it can later be run/trained. On an invalid spec a 422 is returned with a
list of specific problems.
"""

from __future__ import annotations

import asyncio
import threading

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from torch import nn

from .builder import BuildFailed, build_network, count_parameters
from .datasets import (
    BUILTINS,
    DatasetError,
    datasets as dataset_registry,
    load_upload,
    make_builtin,
    match_to_network,
    preview,
)
from .registry import registry
from .runs import runs as run_registry, sse_stream
from .training import run_training
from .train_schema import TrainConfig
from .schema import (
    BuildError,
    BuildFailure,
    BuildSuccess,
    NetworkSpec,
    NodeInfo,
)

# Echo full weight values only for small layers, to bound the response payload.
_WEIGHT_ECHO_LIMIT = 4096
_MAX_UPLOAD = 200 * 1024 * 1024


def _out_dim(built) -> int:
    d = 1
    for s in built.output_shapes[built.module.output_id]:
        d *= s
    return int(d)

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
    with rec.train_lock:
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


# ── Training ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/datasets", tags=["train"])
async def upload_dataset(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > _MAX_UPLOAD:
        return JSONResponse(status_code=422, content={"detail": "file too large (>200MB)"})
    try:
        ds = load_upload(file.filename or "dataset", data)
    except DatasetError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    dataset_registry.add(ds)
    return {"dataset_id": ds.id, **preview(ds)}


@app.post("/api/v1/datasets/builtin", tags=["train"])
def builtin_dataset(name: str, n: int = 1000, noise: float = 0.12, seed: int = 0, classes: int = 2):
    try:
        ds = make_builtin(name, n=n, noise=noise, seed=seed, classes=classes)
    except DatasetError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    dataset_registry.add(ds)
    return {"dataset_id": ds.id, **preview(ds)}


@app.get("/api/v1/datasets/{dataset_id}/preview", tags=["train"])
def dataset_preview(dataset_id: str):
    ds = dataset_registry.get(dataset_id)
    if ds is None:
        return JSONResponse(status_code=404, content={"detail": "dataset not found"})
    return {"dataset_id": ds.id, **preview(ds)}


@app.get("/api/v1/models/{model_id}/datasets", tags=["train"])
def model_datasets(model_id: str):
    rec = registry.get(model_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"detail": "model not found"})
    built = rec.built
    two_d = tuple(built.input_shape) == (2,)
    return {
        "input_shape": list(built.input_shape),
        "out_dim": _out_dim(built),
        "builtins": [{"id": k, "blurb": v} for k, v in BUILTINS.items()] if two_d else [],
    }


@app.post("/api/v1/models/{model_id}/datasets/validate", tags=["train"])
def validate_dataset(model_id: str, body: dict):
    rec = registry.get(model_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"detail": "model not found"})
    ds = dataset_registry.get(body.get("dataset_id", ""))
    if ds is None:
        return JSONResponse(status_code=404, content={"detail": "dataset not found"})
    try:
        X, _y, task, nc = match_to_network(ds, tuple(rec.built.input_shape), _out_dim(rec.built))
    except DatasetError as e:
        return JSONResponse(status_code=422, content={"ok": False, "detail": str(e)})
    return {"ok": True, "task": task, "n_classes": nc, "n_samples": int(X.shape[0]), "out_dim": _out_dim(rec.built)}


@app.post("/api/v1/models/{model_id}/train", tags=["train"])
async def start_training(model_id: str, cfg: TrainConfig):
    rec = registry.get(model_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"detail": "model not found"})
    ds = dataset_registry.get(cfg.dataset_id)
    if ds is None:
        return JSONResponse(status_code=404, content={"detail": "dataset not found"})
    if run_registry.active_for(model_id) is not None:
        return JSONResponse(status_code=409, content={"detail": "a training run is already active for this model"})
    built = rec.built
    out_dim = _out_dim(built)
    try:
        X, y, task, nc = match_to_network(ds, tuple(built.input_shape), out_dim)
    except DatasetError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    loop = asyncio.get_running_loop()
    handle = run_registry.create(model_id, cfg, loop)
    threading.Thread(
        target=run_training,
        args=(handle, rec, built, cfg, X, y, task, nc, out_dim),
        daemon=True,
    ).start()
    return {"run_id": handle.run_id, "status": "running"}


@app.get("/api/v1/runs/{run_id}", tags=["train"])
def run_snapshot(run_id: str):
    run = run_registry.get(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "run not found"})
    return run.snapshot()


@app.get("/api/v1/runs/{run_id}/stream", tags=["train"])
async def run_stream(run_id: str, request: Request):
    run = run_registry.get(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "run not found"})
    lid = request.headers.get("last-event-id", "")
    last = int(lid) if lid.isdigit() else 0
    return StreamingResponse(
        sse_stream(run, last),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def _control(run_id: str, action: str):
    run = run_registry.get(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "run not found"})
    getattr(run, action)()
    return {"status": run.status}


@app.post("/api/v1/runs/{run_id}/pause", tags=["train"])
def run_pause(run_id: str):
    return _control(run_id, "pause")


@app.post("/api/v1/runs/{run_id}/resume", tags=["train"])
def run_resume(run_id: str):
    return _control(run_id, "resume")


@app.post("/api/v1/runs/{run_id}/stop", tags=["train"])
def run_stop(run_id: str):
    return _control(run_id, "stop")


@app.get("/api/v1/runs/{run_id}/checkpoints", tags=["train"])
def run_checkpoints(run_id: str):
    run = run_registry.get(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "run not found"})
    return {"checkpoints": run.snapshot()["checkpoints"]}


@app.post("/api/v1/runs/{run_id}/checkpoints/{ckpt_id}/restore", tags=["train"])
def restore_checkpoint(run_id: str, ckpt_id: str):
    run = run_registry.get(run_id)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "run not found"})
    rec = registry.get(run.model_id)
    if rec is None:
        return JSONResponse(status_code=404, content={"detail": "model not found"})
    if not run.restore(ckpt_id, rec):
        return JSONResponse(status_code=404, content={"detail": "checkpoint not found"})
    return {"ok": True, "ckpt_id": ckpt_id}
