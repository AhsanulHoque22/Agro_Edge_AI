"""
Merge multiple ``training_records.parquet`` files (e.g. synthetic + ThingSpeak).

Assigns fresh ``record_id`` values and optionally shuffles. Does not deduplicate
by timestamp — clean inputs first if needed.

Usage:
  python scripts/merge_training_parquet.py \\
    --input datasets/raw/v1/training_records.parquet \\
    --input datasets/raw/ts_live/training_records.parquet \\
    --dataset-version merged_v2
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, action="append", required=True, help="Parquet file (repeatable).")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet (default: datasets/raw/<dataset-version>/training_records.parquet).",
    )
    p.add_argument(
        "--dataset-version",
        type=str,
        default="merged",
        help="Version written into dataset_version column and default output folder.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffle.")
    p.add_argument("--no-shuffle", action="store_true", help="Keep concat order.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    frames = []
    for ip in args.input:
        path = ip if ip.is_absolute() else root / ip
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path, engine="pyarrow"))

    df = pd.concat(frames, ignore_index=True)
    df["record_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    if "dataset_version" in df.columns:
        df["dataset_version"] = args.dataset_version

    if not args.no_shuffle:
        rng = np.random.default_rng(int(args.seed))
        order = rng.permutation(len(df))
        df = df.iloc[order].reset_index(drop=True)

    out = args.output
    if out is None:
        out = root / "datasets" / "raw" / args.dataset_version / "training_records.parquet"
    else:
        out = out if out.is_absolute() else root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    print(f"Wrote {len(df):,} rows → {out}")


if __name__ == "__main__":
    main()
