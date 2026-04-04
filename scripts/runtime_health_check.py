"""
Health check for AgroEdge runtime scheduler.

Validates that the local JSONL log has a recent successful cycle.
Intended for systemd ExecStartPost, monitoring, or cron.

Exit codes:
  0 — healthy (last line within max age and status ok, or acceptable error with recent attempt)
  1 — unhealthy (missing log, stale timestamp, or repeated failures — optional strict mode)
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_pipeline.jsonl_io import read_last_jsonl_row  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgroEdge runtime log health check.")
    parser.add_argument(
        "--log-path",
        type=str,
        default="logs/runtime_cycles.jsonl",
        help="Path to runtime_cycles JSONL relative to project root or absolute.",
    )
    parser.add_argument(
        "--max-age-seconds",
        type=float,
        default=2700.0,
        help="Max age of last log entry timestamp (default: 2700 = 45 min for 15 min interval + slack).",
    )
    parser.add_argument(
        "--require-ok-status",
        action="store_true",
        help="Fail if last line status is not 'ok' (scheduler still runs but last cycle errored).",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root; default = parent of scripts/.",
    )
    parser.add_argument(
        "--allow-missing-log",
        action="store_true",
        help="Exit 0 if log file is missing or empty (bootstrap / first boot only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.project_root:
        root = Path(args.project_root).resolve()
    else:
        root = Path(__file__).resolve().parent.parent

    log_path = Path(args.log_path)
    if not log_path.is_absolute():
        log_path = root / log_path

    row = read_last_jsonl_row(log_path)
    if row is None:
        if args.allow_missing_log:
            print(f"OK (bootstrap): no log yet: {log_path}")
            sys.exit(0)
        print(f"UNHEALTHY: no log or empty file: {log_path}", file=sys.stderr)
        sys.exit(1)

    ts_raw = row.get("timestamp_utc")
    if not ts_raw:
        print("UNHEALTHY: missing timestamp_utc in last row", file=sys.stderr)
        sys.exit(1)

    try:
        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
    except ValueError as e:
        print(f"UNHEALTHY: bad timestamp: {ts_raw} ({e})", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(UTC)
    age_sec = (now - ts).total_seconds()
    if age_sec > args.max_age_seconds:
        print(
            f"UNHEALTHY: last entry age {age_sec:.0f}s > max {args.max_age_seconds}s "
            f"(timestamp={ts_raw})",
            file=sys.stderr,
        )
        sys.exit(1)

    status = row.get("status")
    if args.require_ok_status and status != "ok":
        print(f"UNHEALTHY: last status={status!r} (require ok)", file=sys.stderr)
        sys.exit(1)

    print(
        f"OK: age={age_sec:.0f}s status={status!r} cycle={row.get('cycle_index')} "
        f"log={log_path}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
