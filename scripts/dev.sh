#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  if [[ -n "${BACK_PID:-}" ]]; then kill "$BACK_PID" 2>/dev/null || true; fi
  if [[ -n "${FRONT_PID:-}" ]]; then kill "$FRONT_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

python -m uvicorn sentinel.main:create_app --factory --reload --host 127.0.0.1 --port 8765 &
BACK_PID=$!

(
  cd apps/frontend
  npm run dev
) &
FRONT_PID=$!

wait