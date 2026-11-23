"""Catalog endpoint + the client/server layer-catalog drift guard.

Adding a layer type touches the backend (``builder.py:LAYERS``) and the frontend
palette (``client/components/build/nodeCatalog.ts:CATALOG``). These two can drift
silently. This test fails loudly if the palette references a type the backend
cannot build, turning a runtime "Unknown layer type" error into a red test.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from tensorcanvas_server.builder import ACTIVATIONS, LAYERS
from tensorcanvas_server.main import app

client = TestClient(app)

# server/tests/ -> server/ -> repo root -> client/components/build/nodeCatalog.ts
_CATALOG_TS = Path(__file__).resolve().parents[2] / "client" / "components" / "build" / "nodeCatalog.ts"


def test_catalog_endpoint_lists_backend_types():
    r = client.get("/api/v1/catalog")
    assert r.status_code == 200
    body = r.json()
    assert set(body["layer_types"]) == set(LAYERS)
    assert set(body["activations"]) == set(ACTIVATIONS)
    # the core drawable layers must be present
    for t in ("input", "output", "linear", "conv2d", "maxpool2d", "flatten"):
        assert t in body["layer_types"]


def test_client_palette_is_subset_of_backend_layers():
    """Every palette type (except the free-form 'custom' escape hatch) must be a
    type the backend can actually build."""
    assert _CATALOG_TS.exists(), f"client catalog not found at {_CATALOG_TS}"
    src = _CATALOG_TS.read_text()
    # Grab the CATALOG array and pull each `type: "..."`.
    block = src.split("export const CATALOG", 1)[1].split("export const CATALOG_BY_TYPE", 1)[0]
    types = set(re.findall(r'type:\s*"([^"]+)"', block))
    assert types, "could not parse any types from nodeCatalog.ts CATALOG"
    drawable = types - {"custom"}
    missing = drawable - set(LAYERS)
    assert not missing, (
        f"client palette references layer types the backend cannot build: {sorted(missing)}. "
        f"Add them to builder.py:LAYERS or remove them from nodeCatalog.ts."
    )
