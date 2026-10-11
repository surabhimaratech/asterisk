#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "$project_dir/backend/.venv/bin/uvicorn" ]]; then
  echo "Backend dependencies are missing. Follow the one-time setup in README.md."
  exit 1
fi

if [[ ! -d "$project_dir/frontend/node_modules" ]]; then
  echo "Frontend dependencies are missing. Follow the one-time setup in README.md."
  exit 1
fi

cleanup() {
  kill "$backend_pid" "$frontend_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(cd "$project_dir/backend" && .venv/bin/uvicorn app.main:app --reload) &
backend_pid=$!
(cd "$project_dir/frontend" && npm run dev) &
frontend_pid=$!

echo "Asterisk is starting at http://127.0.0.1:5173"
wait "$backend_pid" "$frontend_pid"
