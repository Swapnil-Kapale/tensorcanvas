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


def _tiny(linear_params: dict):
    return {
        "nodes": [
            {"id": "in", "type": "input", "params": {"shape": [2]}},
            {"id": "fc", "type": "linear", "params": {"out_features": 2, **linear_params}},
            {"id": "out", "type": "output"},
        ],
        "edges": [
            {"source": "in", "target": "fc"},
            {"source": "fc", "target": "out"},
        ],
    }


def _node(body, nid):
    return next(n for n in body["nodes"] if n["id"] == nid)


def test_bias_disabled_reported():
    r = client.post("/api/v1/models/build", json=_tiny({"bias": False}))
    assert r.status_code == 200, r.text
    fc = _node(r.json(), "fc")
    assert fc["has_bias"] is False
    assert fc["weight_shape"] == [2, 2]
    # 2*2 weights only, no bias params
    assert fc["num_parameters"] == 4


def test_init_weights_applied_and_echoed():
    w = [[1.0, 2.0], [3.0, 4.0]]
    b = [0.5, -0.5]
    r = client.post(
        "/api/v1/models/build",
        json=_tiny({"init_weights": w, "init_bias": b}),
    )
    assert r.status_code == 200, r.text
    fc = _node(r.json(), "fc")
    assert fc["weights"] == w
    assert fc["bias_values"] == b


def test_init_weights_wrong_shape():
    r = client.post(
        "/api/v1/models/build",
        json=_tiny({"init_weights": [[1.0, 2.0, 3.0]]}),  # should be 2x2
    )
    assert r.status_code == 422
    err = r.json()["errors"][0]
    assert err["code"] == "shape_mismatch"
    assert err["node_id"] == "fc"
    assert "init_weights" in err["message"]


def test_activation_attribute_builds():
    r = client.post("/api/v1/models/build", json=_tiny({"activation": "softmax"}))
    assert r.status_code == 200, r.text
    r2 = client.post("/api/v1/models/build", json=_tiny({"activation": "nope"}))
    assert r2.status_code == 422
    assert any("activation" in e["message"] for e in r2.json()["errors"])


def test_per_neuron_activation():
    r = client.post(
        "/api/v1/models/build",
        json=_tiny({"neuron_activations": ["relu", None]}),
    )
    assert r.status_code == 200, r.text
    # wrong length → rejected on that node
    r2 = client.post(
        "/api/v1/models/build",
        json=_tiny({"neuron_activations": ["relu"]}),  # layer has 2 neurons
    )
    assert r2.status_code == 422
    err = r2.json()["errors"][0]
    assert err["node_id"] == "fc"
    assert "neuron_activations" in err["message"]


def test_node_weights_endpoint():
    r = client.post("/api/v1/models/build", json=_tiny({}))
    mid = r.json()["model_id"]
    w = client.get(f"/api/v1/models/{mid}/nodes/fc/weights")
    assert w.status_code == 200, w.text
    body = w.json()
    assert body["weight_shape"] == [2, 2]
    assert len(body["weights"]) == 2 and len(body["weights"][0]) == 2
    # unknown model / node
    assert client.get("/api/v1/models/nope/nodes/fc/weights").status_code == 404
    assert client.get(f"/api/v1/models/{mid}/nodes/in/weights").status_code == 400
