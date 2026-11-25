"""Observe/run: execute the live model on one input, shape every node's activations
into a bounded snapshot (for visualization + analysis), and store snapshots."""

from __future__ import annotations

import base64
import io
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .datasets import DatasetError, datasets, match_to_network
from .training import terminal_activation

ACT_FULL_LIMIT = 4096   # full flat values only when numel <= this
CHANNEL_THUMBS = 8      # conv/image: at most N channel thumbnails
THUMB_HW = 16           # downsample feature maps to 16x16
SNAPSHOT_KEEP = 50      # per-model ring buffer


def _prod(shape) -> int:
    d = 1
    for s in shape:
        d *= int(s)
    return d


def _downsample(ch: torch.Tensor, hw: int) -> list[float]:
    x = ch.reshape(1, 1, *ch.shape).float()
    y = F.adaptive_avg_pool2d(x, (hw, hw)).reshape(hw, hw)
    mx = y.abs().max().clamp_min(1e-9)
    return (y.abs() / mx).reshape(-1).tolist()


def _shape_node(t: torch.Tensor) -> dict:
    a = t.detach()
    flat = a.reshape(-1)
    numel = int(flat.numel())
    absmax = flat.abs().max().clamp_min(1e-9)
    rec: dict = {
        "shape": list(a.shape),
        "summary": {
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "std": float(flat.std(unbiased=False)) if numel > 1 else 0.0,
            "absmax": float(absmax),
            "l2": float(flat.norm()),
            "frac_active": float((flat.abs() > 1e-6).float().mean()),
        },
        "truncated": False,
    }
    if a.dim() <= 1:
        rec["mag"] = (flat.abs() / absmax).tolist()
        rec["values"] = flat.tolist() if numel <= ACT_FULL_LIMIT else None
        rec["truncated"] = numel > ACT_FULL_LIMIT
    elif a.dim() == 3:
        c = a.shape[0]
        cm = a.reshape(c, -1).abs().mean(1)
        rec["mag"] = (cm / cm.max().clamp_min(1e-9)).tolist()
        rec["channels"] = [
            {"c": i, "mean": float(a[i].mean()), "absmax": float(a[i].abs().max())} for i in range(c)
        ]
        rec["thumbs"] = {"hw": THUMB_HW, "images": [_downsample(a[i], THUMB_HW) for i in range(min(c, CHANNEL_THUMBS))]}
        rec["truncated"] = numel > ACT_FULL_LIMIT
    else:
        rec["values"] = flat.tolist() if numel <= ACT_FULL_LIMIT else None
        rec["mag"] = (flat.abs() / absmax).tolist() if numel <= ACT_FULL_LIMIT else None
        rec["truncated"] = numel > ACT_FULL_LIMIT
    return rec


def _predict(out: torch.Tensor, built, target, labels=None) -> dict:
    term = terminal_activation(built)
    o = out.detach().reshape(-1)
    out_dim = int(o.numel())
    if out_dim == 1:
        prob = float(o[0]) if term == "sigmoid" else float(torch.sigmoid(o[0]))
        return {"task": "classification", "pred_class": int(prob > 0.5),
                "probs": [1 - prob, prob], "confidence": max(prob, 1 - prob), "target": target}
    probs = (o if term == "softmax" else torch.softmax(o, 0)).tolist()
    order = sorted(range(out_dim), key=lambda i: probs[i], reverse=True)
    pc = order[0]
    topk = [
        {"index": i, "prob": probs[i], "label": labels[i] if labels and i < len(labels) else None}
        for i in order[:5]
    ]
    return {"task": "classification", "pred_class": int(pc), "probs": probs,
            "confidence": float(probs[pc]), "target": target, "topk": topk}


def _resolve_input(built, spec: dict, input_shape: tuple[int, ...], out_dim: int, norm: dict | None = None):
    kind = spec.get("kind")
    target = None
    task = None
    thumb = None
    if kind == "vector":
        vals = spec.get("values") or []
        if len(vals) != _prod(input_shape):
            raise DatasetError(f"input needs {_prod(input_shape)} values, got {len(vals)}")
        x = torch.tensor(vals, dtype=torch.float32).reshape(1, *input_shape)
    elif kind == "random":
        g = torch.Generator().manual_seed(int(spec.get("seed", 0)))
        x = torch.randn(1, *input_shape, generator=g)
    elif kind == "image":
        if len(input_shape) != 3:
            raise DatasetError("image input needs a 3-D [C, H, W] network input")
        c, h, w = input_shape
        raw = base64.b64decode(str(spec.get("data", "")).split(",")[-1])
        im = Image.open(io.BytesIO(raw)).convert("L" if c == 1 else "RGB").resize((w, h))
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = a[None, :, :] if c == 1 else a.transpose(2, 0, 1)
        x = torch.from_numpy(a).unsqueeze(0)
        thumb = _downsample(x[0, 0], THUMB_HW)
    elif kind == "dataset":
        ds = datasets.get(spec.get("dataset_id", ""))
        if ds is None:
            raise DatasetError("dataset not found")
        X, y, task, _nc = match_to_network(ds, input_shape, out_dim)
        i = int(spec.get("index", 0)) % int(X.shape[0])
        x = X[i : i + 1]
        tv = y[i]
        target = int(tv.item()) if tv.ndim == 0 else tv.tolist()
        if x.dim() == 4:  # image → grayscale thumbnail of channel 0
            thumb = _downsample(x[0, 0], THUMB_HW)
    else:
        raise DatasetError(f"unknown input kind '{kind}'")
    # Pretrained (ImageNet) models expect normalized input. The thumbnail above is
    # built from the raw [0,1] image, so the displayed preview stays natural.
    if norm is not None and len(input_shape) == 3 and x.dim() == 4:
        c = x.shape[1]
        if len(norm.get("mean", [])) == c and len(norm.get("std", [])) == c:
            mean = torch.tensor(norm["mean"], dtype=torch.float32).view(1, c, 1, 1)
            std = torch.tensor(norm["std"], dtype=torch.float32).view(1, c, 1, 1)
            x = (x - mean) / std
    numel = _prod(input_shape)
    record = {
        "source": {"kind": kind, **{k: spec[k] for k in ("dataset_id", "index", "seed") if k in spec}, "target": target},
        "shape": list(input_shape),
        "values": x.reshape(-1).tolist() if numel <= ACT_FULL_LIMIT else None,
        "thumb": thumb,
    }
    return x, record, target, task


def run_model(rec, built, spec: dict) -> dict:
    out_id = built.module.output_id
    out_dim = _prod(built.output_shapes[out_id])
    x, input_rec, target, task = _resolve_input(built, spec, tuple(built.input_shape), out_dim, getattr(rec, "input_norm", None))
    module = built.module
    module.eval()
    with rec.train_lock, torch.no_grad():
        acts = module.forward_capture(x)
    nodes = {nid: _shape_node(acts[nid][0]) for nid in built.order}
    out_t = acts[out_id][0]
    if task == "regression":
        prediction = {"task": "regression", "scalar": out_t.reshape(-1).tolist(), "target": target}
    else:
        prediction = _predict(out_t, built, target, getattr(rec, "labels", None))
    return {
        "weight_version": rec.weight_version,
        "order": list(built.order),
        "node_types": dict(built.node_types),
        "input": input_rec,
        "nodes": nodes,
        "output": {
            "shape": list(out_t.shape),
            "values": out_t.reshape(-1).tolist() if int(out_t.numel()) <= ACT_FULL_LIMIT else None,
            "prediction": prediction,
        },
    }


# ── snapshots ────────────────────────────────────────────────────────────────

@dataclass
class Snapshot:
    snapshot_id: str
    model_id: str
    data: dict
    label: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def meta(self, current_version: int) -> dict:
        pred = self.data["output"]["prediction"]
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at.isoformat(),
            "weight_version": self.data["weight_version"],
            "label": self.label,
            "input_source": self.data["input"]["source"],
            "pred_class": pred.get("pred_class"),
            "confidence": pred.get("confidence"),
            "stale": self.data["weight_version"] != current_version,
        }


class SnapshotRegistry:
    def __init__(self) -> None:
        self._by_id: dict[str, Snapshot] = {}
        self._by_model: dict[str, deque[str]] = {}
        self._lock = threading.Lock()

    def add(self, model_id: str, data: dict, label: Optional[str]) -> Snapshot:
        snap = Snapshot(snapshot_id="snap_" + uuid.uuid4().hex[:12], model_id=model_id, data=data, label=label)
        with self._lock:
            q = self._by_model.setdefault(model_id, deque(maxlen=SNAPSHOT_KEEP))
            if len(q) == q.maxlen:
                self._by_id.pop(q[0], None)  # evicted by append
            q.append(snap.snapshot_id)
            self._by_id[snap.snapshot_id] = snap
        return snap

    def get(self, snapshot_id: str) -> Optional[Snapshot]:
        with self._lock:
            return self._by_id.get(snapshot_id)

    def list_for(self, model_id: str) -> list[Snapshot]:
        with self._lock:
            ids = list(self._by_model.get(model_id, []))
            return [self._by_id[i] for i in reversed(ids) if i in self._by_id]

    def delete(self, snapshot_id: str) -> bool:
        with self._lock:
            snap = self._by_id.pop(snapshot_id, None)
            if snap and snapshot_id in self._by_model.get(snap.model_id, deque()):
                try:
                    self._by_model[snap.model_id].remove(snapshot_id)
                except ValueError:
                    pass
            return snap is not None


snapshots = SnapshotRegistry()
