#!/usr/bin/env bash
# AgroEdge AI — local verification (dependencies + pytest + optional ruff).
# Usage: from repo root: ./scripts/verify.sh
# Environment:
#   VENV — path to virtualenv (default: <root>/.venv)
#   SKIP_RUFF=1 — do not run ruff even if installed

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv}"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install -q -U pip
"$VENV/bin/pip" install -q -r "$ROOT/requirements.txt"

if [[ -f "$ROOT/requirements-dev.txt" ]]; then
  "$VENV/bin/pip" install -q -r "$ROOT/requirements-dev.txt"
fi

echo "==> pytest"
"$VENV/bin/python" -m pytest "$ROOT/tests/" -q

if [[ "${SKIP_RUFF:-0}" == "1" ]]; then
  echo "==> ruff (skipped SKIP_RUFF=1)"
elif [[ -x "$VENV/bin/ruff" ]]; then
  echo "==> ruff check"
  "$VENV/bin/ruff" check "$ROOT"
else
  echo "==> ruff not installed; skip (set SKIP_RUFF=0 and pip install -r requirements-dev.txt)"
fi

echo "verify.sh: OK"
