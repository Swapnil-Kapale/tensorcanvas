"""Datasets for training: user uploads (CSV / JSON / NPY-NPZ / image ZIP), a few
built-in generators, and the logic that matches a dataset to a built network's
input shape and output size.

Design notes:
- Tabular/array datasets are stored as ready tensors (``kind="tensor"``).
- Image ZIPs keep the raw bytes (``kind="images"``) and are decoded + resized to
  the network's (C, H, W) only when matched — so one upload fits any image net.
- ``np.load`` uses ``allow_pickle=False`` and ZIPs are size/entry/path guarded.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import threading
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image

MAX_SAMPLES = 100_000
MAX_ZIP_ENTRIES = 50_000
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


class DatasetError(ValueError):
    """Raised when an upload can't be parsed or matched; message is user-facing."""


@dataclass
class Dataset:
    id: str
    name: str
    format: str  # csv | json | npy | npz | images | builtin
    kind: str  # "tensor" | "images"
    task: str  # "classification" | "regression"
    n_samples: int
    n_classes: Optional[int] = None
    # tensor kind:
    X: Optional[torch.Tensor] = None  # [N, *feature_shape] float32
    feature_shape: Optional[tuple[int, ...]] = None
    # target (both kinds): long [N] for classification, float [N, out] for regression
    y: Optional[torch.Tensor] = None
    # images kind:
    raw_images: Optional[list[bytes]] = None


# ── target / task inference ──────────────────────────────────────────────────

def _targets_from_array(arr: np.ndarray) -> tuple[torch.Tensor, str, Optional[int]]:
    """Turn a target column/array into (tensor, task, n_classes)."""
    a = np.asarray(arr)
    if a.ndim > 1 and a.shape[1] > 1:
        return torch.tensor(a, dtype=torch.float32), "regression", None
    flat = a.reshape(-1).astype(np.float64)
    is_int = np.all(np.isfinite(flat)) and np.allclose(flat, np.round(flat))
    uniq = np.unique(flat)
    # Integer-valued and a small label set → classification.
    if is_int and uniq.size <= max(100, int(flat.size * 0.5)) and uniq.size >= 2:
        mapping = {v: i for i, v in enumerate(uniq.tolist())}
        y = torch.tensor([mapping[v] for v in flat.tolist()], dtype=torch.long)
        return y, "classification", int(uniq.size)
    return torch.tensor(flat, dtype=torch.float32).unsqueeze(1), "regression", None


def _tensor_dataset(X: np.ndarray, y_arr: np.ndarray, name: str, fmt: str) -> Dataset:
    Xa = np.asarray(X, dtype=np.float32)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Xa.shape[0] > MAX_SAMPLES:
        Xa = Xa[:MAX_SAMPLES]
        y_arr = np.asarray(y_arr)[:MAX_SAMPLES]
    if Xa.shape[0] == 0:
        raise DatasetError("dataset is empty")
    y, task, n_classes = _targets_from_array(y_arr)
    if y.shape[0] != Xa.shape[0]:
        raise DatasetError("number of targets does not match number of samples")
    return Dataset(
        id=uuid.uuid4().hex[:12],
        name=name,
        format=fmt,
        kind="tensor",
        task=task,
        n_samples=int(Xa.shape[0]),
        n_classes=n_classes,
        X=torch.from_numpy(Xa),
        feature_shape=tuple(Xa.shape[1:]),
        y=y,
    )


# ── loaders ──────────────────────────────────────────────────────────────────

def load_csv(data: bytes, name: str, target: Optional[int] = None) -> Dataset:
    text = data.decode("utf-8", errors="replace")
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        raise DatasetError("CSV is empty")

    def is_num(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    start = 0 if all(is_num(c) for c in rows[0]) else 1  # skip a header row
    body = rows[start:]
    if not body:
        raise DatasetError("CSV has no data rows")
    width = len(body[0])
    try:
        arr = np.array([[float(c) for c in r] for r in body if len(r) == width], dtype=np.float64)
    except ValueError:
        raise DatasetError(
            "CSV has non-numeric cells; please provide numeric features and labels"
        )
    if arr.shape[1] < 2:
        raise DatasetError("CSV needs at least one feature column and one target column")
    ti = target if target is not None else arr.shape[1] - 1
    if not (0 <= ti < arr.shape[1]):
        raise DatasetError(f"target column {ti} out of range")
    feat = np.delete(arr, ti, axis=1)
    return _tensor_dataset(feat, arr[:, ti], name, "csv")


def load_json(data: bytes, name: str) -> Dataset:
    try:
        obj = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise DatasetError(f"invalid JSON: {e}")
    if isinstance(obj, dict) and obj.get("format") == "tensorcanvas-net":
        raise DatasetError(
            "This is a TensorCanvas network file (.tcnet.json), not a dataset. "
            "Load it on the Build tab via Import or by dragging it onto the canvas."
        )
    if isinstance(obj, dict) and ("X" in obj or "data" in obj):
        X = np.asarray(obj.get("X", obj.get("data")), dtype=np.float64)
        y = np.asarray(obj.get("y", obj.get("labels")))
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        X = np.asarray([row.get("x", row.get("features")) for row in obj], dtype=np.float64)
        y = np.asarray([row.get("y", row.get("label")) for row in obj])
    else:
        raise DatasetError('JSON must be {"X": [...], "y": [...]} or a list of {x, y}')
    return _tensor_dataset(X, y, name, "json")


def load_numpy(data: bytes, name: str) -> Dataset:
    try:
        obj = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as e:  # numpy raises various types
        raise DatasetError(f"could not read numpy file: {e}")
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = obj.files
        xk = "X" if "X" in keys else ("data" if "data" in keys else keys[0])
        yk = "y" if "y" in keys else ("labels" if "labels" in keys else (keys[1] if len(keys) > 1 else None))
        if yk is None:
            raise DatasetError("npz must contain an X and a y array")
        return _tensor_dataset(obj[xk], obj[yk], name, "npz")
    arr = np.asarray(obj)
    if arr.ndim != 2:
        raise DatasetError(".npy must be 2-D [samples, features + target]; use .npz with X and y otherwise")
    return _tensor_dataset(arr[:, :-1], arr[:, -1], name, "npy")


def load_image_zip(data: bytes, name: str) -> Dataset:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise DatasetError("not a valid ZIP file")
    raw: list[bytes] = []
    labels: list[int] = []
    classmap: dict[str, int] = {}
    for info in zf.infolist():
        n = info.filename
        if info.is_dir() or ".." in n or n.startswith("/"):
            continue
        if os.path.splitext(n)[1].lower() not in IMG_EXT:
            continue
        parts = [p for p in n.split("/") if p and not p.startswith(".")]
        if len(parts) < 2:
            continue  # need class-subfolder/image layout
        cls = parts[-2]
        classmap.setdefault(cls, len(classmap))
        raw.append(zf.read(info))
        labels.append(classmap[cls])
        if len(raw) >= MAX_ZIP_ENTRIES or len(raw) >= MAX_SAMPLES:
            break
    if not raw:
        raise DatasetError("ZIP has no images in class subfolders (expected class_name/image.png)")
    if len(classmap) < 2:
        raise DatasetError("image dataset needs at least 2 class folders")
    return Dataset(
        id=uuid.uuid4().hex[:12],
        name=name,
        format="images",
        kind="images",
        task="classification",
        n_samples=len(raw),
        n_classes=len(classmap),
        raw_images=raw,
        y=torch.tensor(labels, dtype=torch.long),
    )


def load_upload(filename: str, data: bytes) -> Dataset:
    ext = os.path.splitext(filename or "")[1].lower()
    name = os.path.basename(filename or "dataset")
    if ext == ".csv":
        return load_csv(data, name)
    if ext == ".json":
        return load_json(data, name)
    if ext in (".npy", ".npz"):
        return load_numpy(data, name)
    if ext == ".zip":
        return load_image_zip(data, name)
    raise DatasetError(f"unsupported file type '{ext}'; use .csv, .json, .npy, .npz or a .zip of images")


# ── matching a dataset to a network ──────────────────────────────────────────

def _check_classification(n_classes: Optional[int], out_dim: int) -> None:
    if n_classes is None:
        raise DatasetError("dataset has no class labels but the network looks like a classifier")
    if out_dim == 1:
        if n_classes != 2:
            raise DatasetError(f"network has 1 output (binary) but the dataset has {n_classes} classes")
    elif n_classes != out_dim:
        raise DatasetError(f"network output has {out_dim} units but the dataset has {n_classes} classes")


def match_to_network(
    ds: Dataset, input_shape: tuple[int, ...], out_dim: int
) -> tuple[torch.Tensor, torch.Tensor, str, Optional[int]]:
    """Return (X, y, task, n_classes) ready to train, or raise DatasetError."""
    if ds.kind == "images":
        if len(input_shape) != 3:
            raise DatasetError(
                f"image dataset needs a 3-D input [C, H, W]; this network's input is {tuple(input_shape)}"
            )
        c, h, w = input_shape
        imgs = []
        for b in ds.raw_images or []:
            im = Image.open(io.BytesIO(b)).convert("L" if c == 1 else "RGB").resize((w, h))
            a = np.asarray(im, dtype=np.float32) / 255.0
            a = a[None, :, :] if c == 1 else a.transpose(2, 0, 1)
            imgs.append(a)
        X = torch.from_numpy(np.stack(imgs)).float()
        _check_classification(ds.n_classes, out_dim)
        return X, ds.y, "classification", ds.n_classes

    assert ds.X is not None and ds.feature_shape is not None
    have = int(np.prod(ds.feature_shape))
    want = int(np.prod(input_shape))
    if have != want:
        raise DatasetError(
            f"dataset has {have} features per sample but the network input needs {want} "
            f"({tuple(input_shape)})"
        )
    X = ds.X.reshape(ds.n_samples, *input_shape)
    if ds.task == "classification":
        _check_classification(ds.n_classes, out_dim)
    else:
        width = ds.y.shape[1] if ds.y.ndim > 1 else 1
        if width != out_dim:
            raise DatasetError(f"regression target has width {width} but the network output is {out_dim}")
    return X, ds.y, ds.task, ds.n_classes


# ── built-in generators (pure torch) ─────────────────────────────────────────

def _gen(name: str, n: int, noise: float, seed: int, classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    k = max(2, classes)
    if name == "moons":
        m = n // 2
        t = torch.linspace(0, math.pi, m)
        a = torch.stack([torch.cos(t), torch.sin(t)], 1)
        b = torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)
        X = torch.cat([a, b]) + torch.randn(2 * m, 2, generator=g) * noise
        y = torch.cat([torch.zeros(m), torch.ones(m)]).long()
        return X, y
    if name == "circles":
        m = n // 2
        t = torch.rand(m, generator=g) * 2 * math.pi
        inner = torch.stack([torch.cos(t), torch.sin(t)], 1) * 0.4
        outer = torch.stack([torch.cos(t), torch.sin(t)], 1) * 1.0
        X = torch.cat([inner, outer]) + torch.randn(2 * m, 2, generator=g) * noise
        y = torch.cat([torch.zeros(m), torch.ones(m)]).long()
        return X, y
    if name == "xor":
        X = torch.rand(n, 2, generator=g) * 2 - 1
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).long()
        X = X + torch.randn(n, 2, generator=g) * noise
        return X, y
    if name == "blobs":
        per = n // k
        cx = [(math.cos(2 * math.pi * i / k) * 2, math.sin(2 * math.pi * i / k) * 2) for i in range(k)]
        Xs, ys = [], []
        for i in range(k):
            pts = torch.randn(per, 2, generator=g) * (noise + 0.25) + torch.tensor(cx[i])
            Xs.append(pts)
            ys.append(torch.full((per,), i))
        return torch.cat(Xs), torch.cat(ys).long()
    # spirals (default)
    per = n // k
    Xs, ys = [], []
    for c in range(k):
        idx = torch.linspace(0, 1, per)
        theta = idx * 3 * math.pi + c * (2 * math.pi / k)
        r = idx
        pts = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], 1)
        pts = pts + torch.randn(per, 2, generator=g) * noise
        Xs.append(pts)
        ys.append(torch.full((per,), c))
    return torch.cat(Xs), torch.cat(ys).long()


BUILTINS = {
    "moons": "Two interleaving half-moons (2 classes)",
    "circles": "Two concentric rings (2 classes)",
    "xor": "XOR quadrants (2 classes)",
    "spirals": "K interleaved spiral arms",
    "blobs": "K Gaussian blobs",
}


def make_builtin(name: str, n: int = 1000, noise: float = 0.12, seed: int = 0, classes: int = 2) -> Dataset:
    if name not in BUILTINS:
        raise DatasetError(f"unknown built-in dataset '{name}'")
    X, y = _gen(name, n, noise, seed, classes)
    nc = int(y.max().item()) + 1
    return Dataset(
        id=uuid.uuid4().hex[:12],
        name=name,
        format="builtin",
        kind="tensor",
        task="classification",
        n_samples=int(X.shape[0]),
        n_classes=nc,
        X=X.float(),
        feature_shape=(2,),
        y=y,
    )


# ── preview (for the UI) ─────────────────────────────────────────────────────

def preview(ds: Dataset, max_points: int = 500) -> dict:
    out: dict = {
        "format": ds.format,
        "kind": ds.kind,
        "task": ds.task,
        "n_samples": ds.n_samples,
        "n_classes": ds.n_classes,
        "feature_shape": list(ds.feature_shape) if ds.feature_shape else None,
    }
    if ds.kind == "tensor" and ds.feature_shape == (2,):
        n = min(ds.n_samples, max_points)
        X = ds.X[:n]
        out["scatter"] = {
            "points": X.tolist(),
            "labels": (ds.y[:n].tolist() if ds.task == "classification" else None),
        }
    elif ds.kind == "images":
        thumbs = []
        for b in (ds.raw_images or [])[:12]:
            im = Image.open(io.BytesIO(b)).convert("L").resize((28, 28))
            thumbs.append((np.asarray(im, dtype=np.float32) / 255.0).reshape(-1).tolist())
        out["thumbs"] = {"size": 28, "images": thumbs, "labels": ds.y[:12].tolist()}
    return out


# ── registry ─────────────────────────────────────────────────────────────────

class DatasetRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Dataset] = {}
        self._lock = threading.Lock()

    def add(self, ds: Dataset) -> str:
        with self._lock:
            self._items[ds.id] = ds
        return ds.id

    def get(self, dataset_id: str) -> Optional[Dataset]:
        with self._lock:
            return self._items.get(dataset_id)


datasets = DatasetRegistry()
