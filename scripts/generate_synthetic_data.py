"""
generate_synthetic_data.py — CLI entry point for synthetic dataset generation.

Generates a labeled TrainingRecord dataset using agronomic irrigation rules,
runs the biological plausibility audit, and saves to Parquet format.

Usage:
    # From agroedge_ai/ directory:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --n-records 5000 --version v2
    python scripts/generate_synthetic_data.py --n-records 20000 --seed 99

Output:
    datasets/raw/<version>/training_records.parquet

The script exits with code 1 if the data audit fails.
No Parquet file is written until the audit passes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from the agroedge_ai/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data_pipeline.feature_engineering.synthetic_generator import SyntheticDataGenerator
from data_pipeline.schemas import GROWTH_STAGE_ENCODING
from data_pipeline.validation.data_audit import DataAudit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AgroEdge AI synthetic irrigation training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-records", type=int, default=20000,
        help="Number of TrainingRecords to generate (default: 20000)",
    )
    parser.add_argument(
        "--version", type=str, default="v1",
        help="Dataset version directory name (default: v1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-farms", type=int, default=15,
        help="Number of simulated farms (default: 15)",
    )
    parser.add_argument(
        "--skip-audit", action="store_true",
        help="Skip the biological plausibility audit (not recommended)",
    )
    return parser.parse_args()


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the generated dataset."""
    irrigated = df[df["irrigation_needed"] == 1]
    positive_rate = df["irrigation_needed"].mean()

    print("\n── Dataset Summary ──────────────────────────────────")
    print(f"  Total records    : {len(df):,}")
    print(f"  Columns          : {len(df.columns)}")
    print(f"  Positive rate    : {positive_rate:.3f} "
          f"({int(positive_rate * len(df)):,} irrigate / "
          f"{int((1 - positive_rate) * len(df)):,} no irrigate)")

    if len(irrigated) > 0:
        print(f"  Mean duration    : {irrigated['irrigation_duration_minutes'].mean():.1f} min "
              f"(irrigated records only)")

    print("\n  Growth stage distribution:")
    stage_name_map = {enc: stage.value for stage, enc in GROWTH_STAGE_ENCODING.items()}
    stage_counts = df["growth_stage_encoded"].value_counts().sort_index()
    for enc, count in stage_counts.items():
        stage_name = stage_name_map.get(int(enc), str(enc))
        bar = "█" * (count // 200)
        pct = count / len(df) * 100
        print(f"    {stage_name:<28} {count:>5}  ({pct:.1f}%)  {bar}")

    print("\n  Per-stage positive rate (irrigation_needed=1):")
    for enc in sorted(stage_counts.index):
        stage_df   = df[df["growth_stage_encoded"] == enc]
        stage_pos  = stage_df["irrigation_needed"].mean()
        stage_name = stage_name_map.get(int(enc), str(enc))
        print(f"    {stage_name:<28} {stage_pos:.3f}")

    print("\n  Sensor value ranges:")
    for col in ["soil_moisture_percent", "air_temperature_celsius",
                "air_humidity_percent", "vpd_kpa"]:
        print(f"    {col:<35} min={df[col].min():.2f}  "
              f"mean={df[col].mean():.2f}  max={df[col].max():.2f}")
    print("─────────────────────────────────────────────────────")


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir   = project_root / "datasets" / "raw" / args.version
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path  = output_dir / "training_records.parquet"

    print("AgroEdge AI — Synthetic Dataset Generator")
    print(f"  Records   : {args.n_records:,}")
    print(f"  Version   : {args.version}")
    print(f"  Seed      : {args.seed}")
    print(f"  Farms     : {args.n_farms}")
    print(f"  Output    : {output_path}")
    print()

    # ── Generate records ──────────────────────────────────────────────────────
    t_start = time.time()
    print("Generating records...")

    generator = SyntheticDataGenerator(
        random_seed=args.seed,
        n_farms=args.n_farms,
        dataset_version=args.version,
    )
    records = generator.generate(n_records=args.n_records)

    t_gen = time.time() - t_start
    print(f"  Done — {len(records):,} records in {t_gen:.1f}s")

    # ── Convert to DataFrame ──────────────────────────────────────────────────
    print("Converting to DataFrame...")
    df = pd.DataFrame([r.model_dump() for r in records])

    # ── Biological plausibility audit ─────────────────────────────────────────
    if not args.skip_audit:
        print("\nRunning biological plausibility audit...")
        audit  = DataAudit()
        report = audit.run(df)
        print()
        print(report.summary())

        if not report.passed:
            print(f"\n  {len(report.failed_checks)} check(s) failed.")
            print("  Dataset rejected. Fix the generator before training.")
            sys.exit(1)

        print()
    else:
        print("WARNING: Audit skipped. Dataset quality is unverified.")

    # ── Print dataset summary ─────────────────────────────────────────────────
    print_dataset_summary(df)

    # ── Save to Parquet ───────────────────────────────────────────────────────
    print("\nSaving to Parquet...")
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    t_total = time.time() - t_start

    print(f"  Saved    : {output_path}")
    print(f"  Size     : {size_mb:.2f} MB")
    print(f"  Total time: {t_total:.1f}s")
    print("\nDataset ready for training.")


if __name__ == "__main__":
    main()
