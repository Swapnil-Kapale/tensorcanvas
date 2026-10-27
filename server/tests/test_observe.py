"""Tests for the Observe/run stage: activation capture, run, snapshots, live patch."""

from __future__ import annotations

import time

import torch
from fastapi.testclient import TestClient

from tensorcanvas_server.main import app
from tensorcanvas_server.builder import build_network
from tensorcanvas_server.schema import NetworkSpec

client = TestClient(app)


def _spec():
    return {
        "nodes": [
            {"id": "in", "type": "input", "params": {"shape": [2]}},
            {"id": "h", "type": "linear", "params": {"out_features": 8, "activation": "relu"}},
            {"id": "o", "type": "linear", "params": {"out_features": 2}},
            {"id": "out", "type": "output", "params": {"activation": "softmax"}},
        ],
        "edges": [
            {"source": "in", "target": "h"},
            {"source": "h", "target": "o"},
            {"source": "o", "target": "out"},
        ],
    }


def _model():
    return client.post("/api/v1/models/build", json=_spec()).json()["model_id"]


def test_forward_capture_matches_forward():
    built = build_network(NetworkSpec(**_spec()))
    x = torch.randn(1, 2)
    acts = built.module.forward_capture(x)
    assert set(acts.keys()) == set(built.order)
    assert torch.allclose(acts[built.module.output_id], built.module(x))


def test_run_vector_input():
    mid = _model()
    r = client.post("/api/v1/models/" + mid + "/run", json={"input": {"kind": "vector", "values": [0.5, -0.3]}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["nodes"].keys()) == {"in", "h", "o", "out"}
    assert body["nodes"]["h"]["shape"] == [8]
    assert len(body["nodes"]["h"]["mag"]) == 8
    pred = body["output"]["prediction"]
    assert pred["task"] == "classification" and pred["pred_class"] in (0, 1)
    assert abs(sum(pred["probs"]) - 1.0) < 1e-4
    assert body["snapshot_id"].startswith("snap_")


def test_run_vector_wrong_size():
    mid = _model()
    r = client.post("/api/v1/models/" + mid + "/run", json={"input": {"kind": "vector", "values": [1, 2, 3]}})
    assert r.status_code == 422


def test_run_dataset_sample_carries_target():
    mid = _model()
    did = client.post("/api/v1/datasets/builtin", params={"name": "xor", "n": 200}).json()["dataset_id"]
    r = client.post("/api/v1/models/" + mid + "/run", json={"input": {"kind": "dataset", "dataset_id": did, "index": 3}})
    assert r.status_code == 200, r.text
    assert r.json()["output"]["prediction"]["target"] in (0, 1)


def test_snapshot_list_get_delete():
    mid = _model()
    sid = client.post("/api/v1/models/" + mid + "/run", json={"input": {"kind": "vector", "values": [0.1, 0.2]}, "label": "t"}).json()["snapshot_id"]
    lst = client.get("/api/v1/models/" + mid + "/snapshots").json()["snapshots"]
    assert any(s["snapshot_id"] == sid for s in lst)
    full = client.get("/api/v1/snapshots/" + sid).json()
    assert full["label"] == "t" and "nodes" in full
    assert client.delete("/api/v1/snapshots/" + sid).json()["ok"] is True
    assert client.get("/api/v1/snapshots/" + sid).status_code == 404


def test_live_patch_changes_weights_and_version():
    mid = _model()
    before = client.get("/api/v1/models/" + mid + "/nodes/o/weights").json()
    new_w = [[0.0] * 8, [0.0] * 8]
    p = client.patch("/api/v1/models/" + mid + "/nodes/o/weights", json={"weights": new_w})
    assert p.status_code == 200, p.text
    assert p.json()["weight_version"] == 1
    after = client.get("/api/v1/models/" + mid + "/nodes/o/weights").json()
    assert after["weights"] == new_w and before["weights"] != new_w
    # a snapshot taken before the patch is now stale
    sid = client.post("/api/v1/models/" + mid + "/run", json={"input": {"kind": "vector", "values": [0.1, 0.2]}}).json()["snapshot_id"]
    client.patch("/api/v1/models/" + mid + "/nodes/o/weights", json={"bias": [0.5, -0.5]})
    meta = [s for s in client.get("/api/v1/models/" + mid + "/snapshots").json()["snapshots"] if s["snapshot_id"] == sid][0]
    assert meta["stale"] is True


def test_live_patch_shape_mismatch():
    mid = _model()
    r = client.patch("/api/v1/models/" + mid + "/nodes/o/weights", json={"weights": [[1, 2, 3]]})
    assert r.status_code == 422


def test_patch_409_during_training():
    mid = _model()
    did = client.post("/api/v1/datasets/builtin", params={"name": "moons", "n": 400}).json()["dataset_id"]
    rid = client.post("/api/v1/models/" + mid + "/train", json={"dataset_id": did, "epochs": 80, "batch_size": 16, "frame_every": 0}).json()["run_id"]
    try:
        p = client.patch("/api/v1/models/" + mid + "/nodes/o/weights", json={"bias": [0.1, 0.1]})
        assert p.status_code == 409
    finally:
        client.post("/api/v1/runs/" + rid + "/stop")
        for _ in range(100):
            if client.get("/api/v1/runs/" + rid).json()["status"] in ("stopped", "completed", "error"):
                break
            time.sleep(0.05)
