"""Pydantic models describing the network-building API contract.

The frontend canvas serialises a neural network as a small graph: a list of
`nodes` (layers / operations) and a list of `edges` (how data flows between
them). This module defines that wire format plus the response shapes.

The graph is intentionally minimal — `in_features` / `in_channels` are *not*
sent by the client; the builder infers them from the graph. See `builder.py`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NodeSpec(BaseModel):
    """A single layer or operation on the canvas."""

    id: str = Field(..., description="Unique node id, as assigned on the canvas.")
    type: str = Field(..., description="Layer/op kind, e.g. 'linear', 'relu'.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific parameters, e.g. {'out_features': 128}.",
    )


class EdgeSpec(BaseModel):
    """A directed connection: output of `source` feeds into `target`."""

    source: str
    target: str


class NetworkSpec(BaseModel):
    """The full network as drawn on the canvas."""

    name: str | None = Field(default=None, description="Optional display name.")
    nodes: list[NodeSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


# ── Responses ────────────────────────────────────────────────────────────────

class NodeInfo(BaseModel):
    """Per-node result reported after a successful build."""

    id: str
    type: str
    output_shape: list[int]  # excludes the batch dimension
    num_parameters: int


class BuildSuccess(BaseModel):
    ok: bool = True
    model_id: str
    name: str | None = None
    num_parameters: int
    nodes: list[NodeInfo]


class BuildError(BaseModel):
    """One problem found in the submitted structure."""

    code: str  # machine-readable, e.g. "unknown_type", "shape_mismatch"
    message: str  # human-readable explanation of what's wrong
    node_id: str | None = None  # the offending node, when applicable


class BuildFailure(BaseModel):
    ok: bool = False
    errors: list[BuildError]
