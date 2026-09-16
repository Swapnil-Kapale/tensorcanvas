#!/usr/bin/env bash
# dev.sh — run the whole TensorCanvas app (client + server) with one command.
#
# Usage:
#   ./dev.sh            start both the frontend and the backend
#
# Stop everything with Ctrl+C — both processes are shut down together.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_DIR="$ROOT/client"
SERVER_DIR="$ROOT/server"

# ── Config (override via env, e.g. `FRONTEND_PORT=4000 ./dev.sh`) ────────────
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

# The backend command. Today the server is just a placeholder (no web
# framework installed yet — we add packages one by one as we need them).
# Once we add FastAPI + uvicorn, switch this to the commented line below.
BACKEND_CMD=(uv run tensorcanvas-server)
# BACKEND_CMD=(uv run uvicorn tensorcanvas_server.main:app --reload --port "$BACKEND_PORT")

# Set this once the backend exposes a health endpoint, to wait for readiness.
BACKEND_HEALTH_URL="${BACKEND_HEALTH_URL:-}"   # e.g. http://localhost:8000/health

# ── Colors ───────────────────────────────────────────────────────────────────
C_WEB="\033[36m"   # cyan
C_API="\033[35m"   # magenta
C_SYS="\033[32m"   # green
C_OFF="\033[0m"

log() { printf "${C_SYS}▶ %s${C_OFF}\n" "$*"; }

# Prefix each line of a child process's output with a colored tag.
prefix() {
  local tag="$1" color="$2"
  while IFS= read -r line; do
    printf "${color}%s${C_OFF} %s\n" "$tag" "$line"
  done
}

# ── Prerequisite checks ───────────────────────────────────────────────────────
command -v uv   >/dev/null 2>&1 || { echo "✗ 'uv' not found. Install: https://docs.astral.sh/uv/"; exit 1; }
command -v yarn >/dev/null 2>&1 || { echo "✗ 'yarn' not found. Run: corepack enable"; exit 1; }

# ── Bootstrap (install deps if missing, so a fresh clone just works) ──────────
if [ ! -d "$CLIENT_DIR/node_modules" ]; then
  log "Installing frontend dependencies (first run)..."
  (cd "$CLIENT_DIR" && yarn install)
fi
if [ ! -d "$SERVER_DIR/.venv" ]; then
  log "Setting up backend environment (first run)..."
  (cd "$SERVER_DIR" && uv sync)
fi

# ── Process management ────────────────────────────────────────────────────────
pids=()
cleanup() {
  echo ""
  log "Stopping servers..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  log "Done."
}
trap cleanup INT TERM EXIT

echo "╔══════════════════════════════════╗"
echo "║        TensorCanvas  ·  dev        ║"
echo "╚══════════════════════════════════╝"

# Start the backend
log "Starting backend (:$BACKEND_PORT)..."
( cd "$SERVER_DIR" && "${BACKEND_CMD[@]}" 2>&1 | prefix "[api]" "$C_API" ) &
pids+=($!)

# Optionally wait for the backend to report healthy before starting the client
if [ -n "$BACKEND_HEALTH_URL" ]; then
  printf "  waiting for backend"
  for _ in $(seq 1 20); do
    if curl -sf "$BACKEND_HEALTH_URL" >/dev/null 2>&1; then printf " ✓\n"; break; fi
    printf "."; sleep 0.5
  done
fi

# Start the frontend
log "Starting frontend (:$FRONTEND_PORT)..."
( cd "$CLIENT_DIR" && yarn dev --port "$FRONTEND_PORT" 2>&1 | prefix "[web]" "$C_WEB" ) &
pids+=($!)

echo ""
printf "  ${C_WEB}frontend${C_OFF} → http://localhost:%s\n" "$FRONTEND_PORT"
printf "  ${C_API}backend ${C_OFF} → http://localhost:%s\n" "$BACKEND_PORT"
echo ""
echo "  Press Ctrl+C to stop both."
echo ""

# Wait until a process exits (or Ctrl+C triggers cleanup)
wait
