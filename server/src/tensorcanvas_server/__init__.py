"""TensorCanvas backend package.

``main()`` is the console-script entry point (``uv run tensorcanvas-server``).
It launches the FastAPI app with uvicorn so the same command that scaffolding
created now runs the real server — which is what ``dev.sh`` invokes.
"""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("BACKEND_PORT", os.environ.get("PORT", "8000")))
    reload = os.environ.get("RELOAD", "1") != "0"
    uvicorn.run("tensorcanvas_server.main:app", host=host, port=port, reload=reload)
