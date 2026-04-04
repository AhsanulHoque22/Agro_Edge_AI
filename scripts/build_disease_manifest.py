"""
CLI for building the disease dataset manifest.parquet.

Usage (from `agroedge_ai/`):

  python scripts/build_disease_manifest.py \
    --dataset-root datasets/disease_model/v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.disease.manifest_builder import build_disease_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets") / "disease_model" / "v1",
        help="Path to datasets/disease_model/<version>.",
    )
    p.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap per class for quick experiments.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output = build_disease_manifest(
        dataset_root=dataset_root,
        limit_per_class=args.limit_per_class,
    )
    print(f"Built manifest: {output}")


if __name__ == "__main__":
    main()

