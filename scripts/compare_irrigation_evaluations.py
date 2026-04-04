"""
Compare two irrigation evaluation.json reports and print metric deltas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True, help="Path to baseline evaluation.json")
    p.add_argument("--candidate", type=Path, required=True, help="Path to candidate evaluation.json")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON report path.",
    )
    return p.parse_args()


def _get(d: dict, *keys: str):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _delta(candidate, baseline):
    if candidate is None or baseline is None:
        return None
    try:
        return float(candidate) - float(baseline)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    b = json.loads(args.baseline.read_text(encoding="utf-8"))
    c = json.loads(args.candidate.read_text(encoding="utf-8"))

    pairs = {
        "classifier_f1_score": (
            _get(b, "classifier_metrics", "f1_score"),
            _get(c, "classifier_metrics", "f1_score"),
        ),
        "classifier_accuracy": (
            _get(b, "classifier_metrics", "accuracy"),
            _get(c, "classifier_metrics", "accuracy"),
        ),
        "regressor_r2_score": (
            _get(b, "regressor_metrics", "r2_score"),
            _get(c, "regressor_metrics", "r2_score"),
        ),
        "regressor_mae_minutes": (
            _get(b, "regressor_metrics", "mae_minutes"),
            _get(c, "regressor_metrics", "mae_minutes"),
        ),
        "regressor_mae_minutes_when_irrigating": (
            _get(b, "regressor_metrics", "mae_minutes_when_irrigating"),
            _get(c, "regressor_metrics", "mae_minutes_when_irrigating"),
        ),
    }

    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "comparison": {},
    }
    print("Irrigation evaluation comparison")
    for name, (b_val, c_val) in pairs.items():
        d = _delta(c_val, b_val)
        report["comparison"][name] = {
            "baseline": b_val,
            "candidate": c_val,
            "delta_candidate_minus_baseline": d,
        }
        print(f"  {name}: baseline={b_val} candidate={c_val} delta={d}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote comparison report: {args.output}")


if __name__ == "__main__":
    main()
