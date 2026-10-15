"""Tests for the training stage: datasets, matching, and the train loop."""

from __future__ import annotations

import io
import time

import torch
from fastapi.testclient import TestClient

from tensorcanvas_server.main import app
from tensorcanvas_server import datasets as ds

client = TestClient(app)


def _build_2in_2class():
    spec = {
        "nodes": [
            {"id": "in", "type": "input", "params": {"shape": [2]}},
            {"id": "h", "type": "linear", "params": {"out_features": 16, "activation": "relu"}},
            {"id": "out_fc", "type": "linear", "params": {"out_features": 2}},
            {"id": "out", "type": "output"},
        ],
        "edges": [
            {"source": "in", "target": "h"},
            {"source": "h", "target": "out_fc"},
            {"source": "out_fc", "target": "out"},
        ],
    }
    r = client.post("/api/v1/models/build", json=spec)
    assert r.status_code == 200, r.text
    return r.json()["model_id"]


def _wait(run_id, timeout=40.0):
    losses = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = client.get(f"/api/v1/runs/{run_id}").json()
        if snap.get("latest", {}).get("train_loss") is not None:
            losses.append(snap["latest"]["train_loss"])
        if snap["status"] in ("completed", "stopped", "error"):
            return snap, losses
        time.sleep(0.1)
    return client.get(f"/api/v1/runs/{run_id}").json(), losses


# ── dataset matching (unit) ──────────────────────────────────────────────────

def test_match_shape_mismatch():
    d = ds.make_builtin("xor", n=50)  # 2 features
    import pytest

    with pytest.raises(ds.DatasetError):
        ds.match_to_network(d, (3,), 2)  # network wants 3 features


def test_csv_loader_infers_classification():
    rows = ["0.1,0.2,0", "0.9,0.8,1", "0.2,0.1,0", "0.8,0.9,1"]
    d = ds.load_csv(("\n".join(rows)).encode(), "d.csv")
    assert d.task == "classification"
    assert d.n_classes == 2
    assert d.feature_shape == (2,)


# ── endpoints + training ─────────────────────────────────────────────────────

def test_builtin_and_validate():
    mid = _build_2in_2class()
    b = client.post("/api/v1/datasets/builtin", params={"name": "xor", "n": 400})
    assert b.status_code == 200, b.text
    did = b.json()["dataset_id"]
    v = client.post(f"/api/v1/models/{mid}/datasets/validate", json={"dataset_id": did})
    assert v.status_code == 200, v.text
    assert v.json()["ok"] is True and v.json()["task"] == "classification"


def test_csv_upload_endpoint():
    mid = _build_2in_2class()
    csv_bytes = "\n".join(f"{i*0.01},{i*0.02},{i % 2}" for i in range(20)).encode()
    up = client.post("/api/v1/datasets", files={"file": ("d.csv", csv_bytes, "text/csv")})
    assert up.status_code == 200, up.text
    v = client.post(f"/api/v1/models/{mid}/datasets/validate", json={"dataset_id": up.json()["dataset_id"]})
    assert v.status_code == 200, v.text


def test_train_runs_and_mutates_weights():
    mid = _build_2in_2class()
    before = client.get(f"/api/v1/models/{mid}/nodes/h/weights").json()["weights"]
    did = client.post("/api/v1/datasets/builtin", params={"name": "xor", "n": 600}).json()["dataset_id"]
    cfg = {
        "dataset_id": did, "epochs": 30, "batch_size": 64, "lr": 0.05,
        "optimizer": "adam", "log_every": 1, "frame_every": 0, "val_split": 0.2,
    }
    r = client.post(f"/api/v1/models/{mid}/train", json=cfg)
    assert r.status_code == 200, r.text
    snap, losses = _wait(r.json()["run_id"])
    assert snap["status"] == "completed", snap
    assert snap["latest"]["val_acc"] is not None
    assert snap["latest"]["val_acc"] >= 0.5
    after = client.get(f"/api/v1/models/{mid}/nodes/h/weights").json()["weights"]
    diff = (torch.tensor(after) - torch.tensor(before)).abs().sum().item()
    assert diff > 1e-4  # weights actually changed
    if len(losses) >= 2:
        assert losses[-1] <= losses[0] + 1e-6  # loss did not increase


def test_train_409_when_active():
    mid = _build_2in_2class()
    did = client.post("/api/v1/datasets/builtin", params={"name": "spirals", "n": 400, "classes": 2}).json()["dataset_id"]
    cfg = {"dataset_id": did, "epochs": 50, "batch_size": 32, "frame_every": 0}
    r1 = client.post(f"/api/v1/models/{mid}/train", json=cfg)
    assert r1.status_code == 200
    r2 = client.post(f"/api/v1/models/{mid}/train", json=cfg)
    assert r2.status_code == 409
    client.post(f"/api/v1/runs/{r1.json()['run_id']}/stop")
    _wait(r1.json()["run_id"])


def test_checkpoint_restore():
    mid = _build_2in_2class()
    did = client.post("/api/v1/datasets/builtin", params={"name": "moons", "n": 400}).json()["dataset_id"]
    cfg = {"dataset_id": did, "epochs": 6, "batch_size": 64, "frame_every": 0, "log_every": 1}
    rid = client.post(f"/api/v1/models/{mid}/train", json=cfg).json()["run_id"]
    snap, _ = _wait(rid)
    cks = client.get(f"/api/v1/runs/{rid}/checkpoints").json()["checkpoints"]
    assert len(cks) >= 1
    rr = client.post(f"/api/v1/runs/{rid}/checkpoints/{cks[0]['ckpt_id']}/restore")
    assert rr.status_code == 200 and rr.json()["ok"] is True
