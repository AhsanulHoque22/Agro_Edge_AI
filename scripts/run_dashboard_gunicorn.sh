#!/usr/bin/env bash
# Run the monitoring dashboard under Gunicorn (multi-worker WSGI).
# Requires: pip install -r requirements.txt -r requirements-prod.txt
#
# Environment (optional):
#   FLASK_RUN_HOST   — bind address (default 0.0.0.0)
#   FLASK_RUN_PORT   — port (default 5000)
#   GUNICORN_WORKERS — worker processes (default 2)
#   GUNICORN_THREADS — threads per worker (default 2)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${FLASK_RUN_HOST:-0.0.0.0}"
PORT="${FLASK_RUN_PORT:-5000}"
WORKERS="${GUNICORN_WORKERS:-2}"
THREADS="${GUNICORN_THREADS:-2}"

exec gunicorn \
  --chdir "$ROOT" \
  --bind "${HOST}:${PORT}" \
  --workers "$WORKERS" \
  --threads "$THREADS" \
  --access-logfile "-" \
  --error-logfile "-" \
  --capture-output \
  web_dashboard.app:app
