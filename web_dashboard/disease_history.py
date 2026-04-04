"""Disease inference history helpers.

Predictions are appended to a JSONL log so the dashboard can show the latest
result even after process restarts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from data_pipeline.jsonl_io import read_last_jsonl_row, read_last_jsonl_rows


def _resolve_disease_log_path(project_root: Path) -> Path:
    raw = os.getenv("AGROEDGE_DISEASE_LOG", "logs/disease_predictions.jsonl")
    p = Path(raw)
    if not p.is_absolute():
        p = project_root / p
    return p


def append_disease_prediction(project_root: Path, row: dict[str, Any]) -> Path:
    """Append one JSON object to the disease JSONL log."""
    path = _resolve_disease_log_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        import json

        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_last_disease_prediction(project_root: Path) -> tuple[dict[str, Any] | None, Path]:
    path = _resolve_disease_log_path(project_root)
    return read_last_jsonl_row(path), path


def read_last_disease_predictions(
    project_root: Path, *, limit: int = 5
) -> tuple[list[dict[str, Any]], Path]:
    path = _resolve_disease_log_path(project_root)
    rows = read_last_jsonl_rows(path, limit=limit)
    return rows, path


def register_disease_history_routes(app, project_root: Path) -> None:
    from flask import Response, jsonify

    @app.get("/api/disease/last")
    def api_disease_last() -> tuple[Response, int]:
        row, path = read_last_disease_prediction(project_root)
        if row is None:
            return (
                jsonify({"ok": False, "error": "no_disease_log", "path": str(path)}),
                503,
            )
        return jsonify({"ok": True, "path": str(path), "row": row}), 200

    @app.get("/api/disease/history")
    def api_disease_history() -> tuple[Any, int]:
        from flask import request

        limit_raw = request.args.get("limit", "")
        try:
            limit = int(limit_raw) if limit_raw else 5
        except ValueError:
            limit = 5
        limit = max(1, min(limit, 50))

        rows, path = read_last_disease_predictions(project_root, limit=limit)
        if not rows:
            return jsonify({"ok": False, "error": "no_disease_log", "path": str(path)}), 503
        return jsonify({"ok": True, "path": str(path), "rows": rows}), 200
