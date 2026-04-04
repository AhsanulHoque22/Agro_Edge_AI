"""
Smoke test: load disease bundle and run a few random feature vectors.

Requires an export from ``scripts/train_disease_model.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge_inference.disease_engine import DiseaseInferenceEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bundle-version",
        type=str,
        default="v0.1.0",
        help="Directory under model_export/disease_model/ (default: v0.1.0).",
    )
    p.add_argument("--samples", type=int, default=3, help="Random vectors to classify.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    bundle_dir = root / "model_export" / "disease_model" / args.bundle_version
    if not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"Bundle not found: {bundle_dir}. Run: python scripts/train_disease_model.py"
        )

    engine = DiseaseInferenceEngine(bundle_dir=bundle_dir)
    rng = np.random.default_rng(int(args.seed))
    dim = engine.feature_dimension
    if dim <= 0:
        raise ValueError("Bundle missing feature_dimension in model.joblib payload")

    print("Disease inference smoke test")
    print(f"  bundle: {bundle_dir}")
    print(f"  feature_dim: {dim}")
    print(f"  classes: {engine.class_names}")
    print()

    for i in range(int(args.samples)):
        vec = rng.normal(0.0, 1.0, size=dim)
        out = engine.predict_vector(vec)
        payload = {
            "sample": i,
            "predicted": out.disease_class_name,
            "index": out.disease_class_index,
            "model_version": out.model_version,
        }
        if out.probabilities:
            payload["top_prob"] = round(max(out.probabilities.values()), 4)
        print(json.dumps(payload, indent=2))
        print()


if __name__ == "__main__":
    main()
