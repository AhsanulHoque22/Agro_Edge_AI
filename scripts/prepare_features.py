"""
CLI to prepare model-ready features from raw training records.

Usage:
  python scripts/prepare_features.py --version v1
  python scripts/prepare_features.py --version v1_1 \\
    --feature-schema-path configs/feature_schema_v1_1.yaml \\
    --processed-version v1_1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow script execution from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.feature_engineering.prepare_features import FeaturePreparer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AgroEdge irrigation model feature dataset."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Dataset version to process (default: v1).",
    )
    parser.add_argument(
        "--raw-parquet",
        type=Path,
        default=None,
        help="Override path to training_records.parquet (default: datasets/raw/<version>/).",
    )
    parser.add_argument(
        "--feature-schema-path",
        type=Path,
        default=None,
        help="YAML feature contract (default: configs/feature_schema.yaml).",
    )
    parser.add_argument(
        "--processed-version",
        type=str,
        default=None,
        help="Output directory under datasets/processed/ (default: same as --version).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    raw_path = args.raw_parquet
    if raw_path is None:
        raw_path = root / "datasets" / "raw" / args.version / "training_records.parquet"
    else:
        raw_path = raw_path if raw_path.is_absolute() else root / raw_path
    proc_ver = args.processed_version or args.version
    out_dir = root / "datasets" / "processed" / proc_ver
    schema_path = (
        (args.feature_schema_path if args.feature_schema_path.is_absolute() else root / args.feature_schema_path)
        if args.feature_schema_path is not None
        else root / "configs" / "feature_schema.yaml"
    )

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    print(f"Loading raw dataset: {raw_path}")
    raw_df = pd.read_parquet(raw_path, engine="pyarrow")
    print(f"  rows={len(raw_df):,} cols={len(raw_df.columns)}")

    preparer = FeaturePreparer(feature_schema_path=schema_path)
    prepared = preparer.prepare(raw_df)
    result = preparer.save(prepared, out_dir)

    print("\nPrepared dataset saved:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
