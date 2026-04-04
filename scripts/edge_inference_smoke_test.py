"""
Smoke test for edge inference using exported model bundle and processed samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge_inference import EdgeInferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgroEdge edge inference smoke test.")
    parser.add_argument(
        "--bundle-version",
        type=str,
        default="v1.0.0",
        help="Exported model bundle version directory name (default: v1.0.0).",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="v1",
        help="Processed dataset version for sample rows (default: v1).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of rows to test (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    bundle_dir = root / "model_export" / "irrigation_model" / args.bundle_version
    x_path = root / "datasets" / "processed" / args.dataset_version / "X_features.parquet"

    if not bundle_dir.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_dir}")
    if not x_path.exists():
        raise FileNotFoundError(f"Processed features not found: {x_path}")

    engine = EdgeInferenceEngine(bundle_dir=bundle_dir)
    X = pd.read_parquet(x_path, engine="pyarrow")
    n = min(args.samples, len(X))
    sample_df = X.head(n)

    print("Edge inference smoke test")
    print(f"  bundle: {bundle_dir}")
    print(f"  samples: {n}")
    print()

    for idx, row in sample_df.iterrows():
        payload = row.to_dict()
        decision = engine.decide(payload)
        print(f"[sample {idx}]")
        print(
            json.dumps(
                {
                    "should_irrigate": decision.should_irrigate,
                    "approved_duration_minutes": round(decision.approved_duration_minutes, 2),
                    "blocked_reason": decision.blocked_reason,
                    "irrigation_probability": round(decision.model_prediction.irrigation_probability, 4),
                    "predicted_duration": round(decision.model_prediction.irrigation_duration_minutes, 2),
                    "model_version": decision.model_version,
                },
                indent=2,
            )
        )
        print()


if __name__ == "__main__":
    main()
