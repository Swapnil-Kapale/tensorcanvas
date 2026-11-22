"""In-memory store of built models.

A successful build is kept here keyed by a generated ``model_id`` so that
future endpoints (run inference, train, inspect activations) can look the live
``nn.Module`` back up without rebuilding it. This is process-local and not
persisted — fine for the dev/studio workflow; swappable for real storage later.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .builder import BuiltModel


@dataclass
class ModelRecord:
    model_id: str
    built: BuiltModel
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Guards weight reads (e.g. GET /weights) against in-place optimizer.step().
    train_lock: threading.Lock = field(default_factory=threading.Lock)
    # Bumped on every in-place weight mutation (training step, live patch, checkpoint
    # restore) so snapshots taken earlier can be flagged stale.
    weight_version: int = 0
    # Class names for the output units (set when pretrained weights are loaded).
    labels: Optional[list[str]] = None


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelRecord] = {}
        self._lock = threading.Lock()

    def add(self, built: BuiltModel) -> str:
        model_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._models[model_id] = ModelRecord(model_id=model_id, built=built)
        return model_id

    def get(self, model_id: str) -> ModelRecord | None:
        with self._lock:
            return self._models.get(model_id)

    def count(self) -> int:
        with self._lock:
            return len(self._models)


# A single shared registry for the app.
registry = ModelRegistry()
