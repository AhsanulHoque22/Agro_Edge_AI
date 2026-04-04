"""Small JSONL helpers shared by health checks and local dashboards."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


def read_last_jsonl_row(path: Path) -> dict[str, Any] | None:
    """Return the last non-empty JSON object in a JSONL file, or None if unreadable."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    last_line = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        return None
    return json.loads(last_line)


def read_last_jsonl_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    """
    Return up to ``limit`` last non-empty JSON objects from a JSONL file.

    Order is newest-first (last line becomes index 0).
    """
    if limit <= 0:
        return []
    if not path.exists() or path.stat().st_size == 0:
        return []

    dq: deque[dict[str, Any]] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dq.append(json.loads(line))

    # deque holds oldest->newest; return newest->oldest.
    return list(reversed(list(dq)))
