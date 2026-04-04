"""
Build a fixed validation dataset from raw training records.

Deterministic split using stable hash of record_id, so future runs can compare
model versions against the same validation slice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input parquet (typically datasets/raw/<version>/training_records.parquet).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("datasets") / "validation" / "fixed_v1" / "validation_records.parquet",
        help="Output parquet for fixed validation slice.",
    )
    p.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of rows assigned to validation (0,1).",
    )
    p.add_argument(
        "--salt",
        type=str,
        default="agroedge_fixed_validation_v1",
        help="Hash salt to freeze/rotate validation cohorts intentionally.",
    )
    return p.parse_args()


def _score(value: str) -> float:
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()
    # 16 hex chars -> deterministic 64-bit bucket
    n = int(h[:16], 16)
    return n / float(0xFFFFFFFFFFFFFFFF)


def main() -> None:
    args = parse_args()
    if not (0.0 < float(args.holdout_ratio) < 1.0):
        raise ValueError("--holdout-ratio must be between 0 and 1")

    in_path = args.input
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    df = pd.read_parquet(in_path, engine="pyarrow")
    if "record_id" not in df.columns:
        raise ValueError("Input parquet missing required column: record_id")

    token = args.salt.strip()
    if not token:
        raise ValueError("--salt must be non-empty")

    scores = df["record_id"].astype(str).map(lambda x: _score(f"{token}:{x}"))
    val_df = df.loc[scores < float(args.holdout_ratio)].copy()
    train_df = df.loc[scores >= float(args.holdout_ratio)].copy()

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    val_df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")

    meta = {
        "input": str(in_path),
        "output": str(out),
        "holdout_ratio": float(args.holdout_ratio),
        "salt": token,
        "n_total": int(len(df)),
        "n_validation": int(len(val_df)),
        "n_non_validation": int(len(train_df)),
        "class_distribution_validation": (
            val_df["irrigation_needed"].value_counts(dropna=False).to_dict()
            if "irrigation_needed" in val_df.columns
            else {}
        ),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote fixed validation set: {out}")
    print(f"Wrote metadata: {meta_path}")
    print(
        f"Rows total={meta['n_total']} validation={meta['n_validation']} "
        f"non_validation={meta['n_non_validation']}"
    )


if __name__ == "__main__":
    main()
