"""Tests for the network build endpoint and the builder's validation."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tensorcanvas_server.main import app

client = TestClient(app)


def _mlp():
    return {
        "name": "mnist-mlp",
        "nodes": [
            {"id": "in", "type": "input", "params": {"shape": [1, 28, 28]}},
            {"id": "flat", "type": "flatten"},
            {"id": "fc1", "type": "linear", "params": {"out_features": 128}},
            {"id": "act1", "type": "relu"},
            {"id": "fc2", "type": "linear", "params": {"out_features": 10}},
            {"id": "out", "type": "output"},
        ],
        "edges": [
            {"source": "in", "target": "flat"},
            {"source": "flat", "target": "fc1"},
            {"source": "fc1", "target": "act1"},
            {"source": "act1", "target": "fc2"},
            {"source": "fc2", "target": "out"},
        ],
    }


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_build_valid_mlp():
    r = client.post("/api/v1/models/build", json=_mlp())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["model_id"]
    # 784*128 + 128  +  128*10 + 10
    assert body["num_parameters"] == 784 * 128 + 128 + 128 * 10 + 10
    shapes = {n["id"]: n["output_shape"] for n in body["nodes"]}
    assert shapes["flat"] == [784]
    assert shapes["fc1"] == [128]
    assert shapes["out"] == [10]


def test_build_valid_cnn():
    spec = {
        "nodes": [
            {"id": "in", "type": "input", "params": {"shape": [1, 28, 28]}},
            {"id": "c1", "type": "conv2d",
             "params": {"out_channels": 8, "kernel_size": 3}},
            {"id": "p1", "type": "maxpool2d", "params": {"kernel_size": 2}},
            {"id": "flat", "type": "flatten"},
            {"id": "fc", "type": "linear", "params": {"out_features": 10}},
            {"id": "out", "type": "output"},
        ],
        "edges": [
            {"source": "in", "target": "c1"},
            {"source": "c1", "target": "p1"},
            {"source": "p1", "target": "flat"},
            {"source": "flat", "target": "fc"},
            {"source": "fc", "target": "out"},
        ],
    }
    r = client.post("/api/v1/models/build", json=spec)
    assert r.status_code == 200, r.text
    shapes = {n["id"]: n["output_shape"] for n in r.json()["nodes"]}
    assert shapes["c1"] == [8, 26, 26]      # 28-3+1
    assert shapes["p1"] == [8, 13, 13]      # 26 / 2
    assert shapes["flat"] == [8 * 13 * 13]


def test_linear_without_flatten_reports_shape_error():
    spec = {
        "nodes": [
            {"id": "in", "type": "input", "params": {"shape": [1, 28, 28]}},
            {"id": "fc", "type": "linear", "params": {"out_features": 10}},
            {"id": "out", "type": "output"},
        ],
        "edges": [
            {"source": "in", "target": "fc"},
            {"source": "fc", "target": "out"},
        ],
    }
    r = client.post("/api/v1/models/build", json=spec)
    assert r.status_code == 422
    errs = r.json()["errors"]
    assert errs[0]["code"] == "shape_mismatch"
    assert errs[0]["node_id"] == "fc"
    assert "flatten" in errs[0]["message"].lower()


def test_unknown_type():
    spec = {
        "nodes": [
            {"id": "in", "type": "input", "params": {"shape": [4]}},
            {"id": "x", "type": "transmogrify"},
            {"id": "out", "type": "output"},
        ],
        "edges": [
            {"source": "in", "target": "x"},
            {"source": "x", "target": "out"},
        ],
    }
    r = client.post("/api/v1/models/build", json=spec)
    assert r.status_code == 422
    codes = {e["code"] for e in r.json()["errors"]}
    assert "unknown_type" in codes


def test_missing_out_features():
    spec = {
        "nodes": [
            {"id": "in", "type": "input", "params": {"shape": [4]}},
            {"id": "fc", "type": "linear"},
            {"id": "out", "type": "output"},
        ],
        "edges": [
            {"source": "in", "target": "fc"},
            {"source": "fc", "target": "out"},
        ],
    }
    r = client.post("/api/v1/models/build", json=spec)
    assert r.status_code == 422
    assert any("out_features" in e["message"] for e in r.json()["errors"])


def test_two_inputs_rejected():
    spec = {
        "nodes": [
            {"id": "in1", "type": "input", "params": {"shape": [4]}},
            {"id": "in2", "type": "input", "params": {"shape": [4]}},
            {"id": "out", "type": "output"},
        ],
        "edges": [
            {"source": "in1", "target": "out"},
            {"source": "in2", "target": "out"},
        ],
    }
    r = client.post("/api/v1/models/build", json=spec)
    assert r.status_code == 422
    assert any(e["code"] == "input_count" for e in r.json()["errors"])


def test_disconnected_node():
    spec = {
        "nodes": [
            {"id": "in", "type": "input", "params": {"shape": [4]}},
            {"id": "fc", "type": "linear", "params": {"out_features": 2}},
            {"id": "lonely", "type": "relu"},
            {"id": "out", "type": "output"},
        ],
        "edges": [
            {"source": "in", "target": "fc"},
            {"source": "fc", "target": "out"},
            {"source": "lonely", "target": "lonely"},
        ],
    }
    r = client.post("/api/v1/models/build", json=spec)
    assert r.status_code == 422
    # a self-loop on 'lonely' shows up as a bad-degree / connectivity problem
    assert r.json()["ok"] is False
