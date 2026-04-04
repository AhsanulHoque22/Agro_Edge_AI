"""
Pull ThingSpeak channel history and write ``TrainingRecord`` Parquet for retraining.

Loads API keys from the environment (optionally ``python-dotenv`` from ``.env``).

Usage (from ``agroedge_ai/``):

  python scripts/pull_thingspeak_training_data.py \\
    --days 90 \\
    --farm-id farm_001 --field-id field_001 --node-id node_esp32_001 \\
    --planting-date 2024-06-01 \\
    --output datasets/raw/ts_pull/training_records.parquet

Then run feature prep + training as with synthetic data (different path/version).

Negatives (optional): ``--negatives-ratio 1.0`` adds env samples not near an
irrigation event (see ``--association-hours``).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from dotenv import load_dotenv

from data_pipeline.ingestion import (
    build_training_records,
    fetch_feeds_timerange,
)
from data_pipeline.validation.data_audit import DataAudit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default datasets/raw/<version>/training_records.parquet)",
    )
    p.add_argument("--version", type=str, default="ts_field", help="Dataset version tag / folder name")
    p.add_argument("--days", type=float, default=90.0, help="How far back from now to fetch (default 90)")
    p.add_argument(
        "--range-start",
        type=str,
        default=None,
        help="ISO start datetime (overrides --days if both --range-end set)",
    )
    p.add_argument("--range-end", type=str, default=None, help="ISO end datetime (default: now UTC)")
    p.add_argument("--chunk-days", type=float, default=21.0, help="ThingSpeak fetch chunk size (default 21)")
    p.add_argument("--pause-seconds", type=float, default=1.0, help="Pause between ThingSpeak requests")
    p.add_argument("--max-feeds-per-channel", type=int, default=None, help="Cap rows per channel after merge")
    p.add_argument("--base-url", type=str, default=None, help="Default THINGSPEAK_BASE_URL from env")
    p.add_argument("--env-channel-id", type=str, default=None, help="Default THINGSPEAK_ENV_CHANNEL_ID")
    p.add_argument("--env-read-key", type=str, default=None, help="Default THINGSPEAK_ENV_READ_API_KEY")
    p.add_argument("--irr-channel-id", type=str, default=None, help="Default THINGSPEAK_IRRIGATION_CHANNEL_ID")
    p.add_argument("--irr-read-key", type=str, default=None, help="Default THINGSPEAK_IRRIGATION_READ_API_KEY")
    p.add_argument("--farm-id", type=str, required=True)
    p.add_argument("--field-id", type=str, required=True)
    p.add_argument("--node-id", type=str, required=True)
    p.add_argument("--planting-date", type=str, required=True, help="ISO date YYYY-MM-DD")
    p.add_argument("--harvest-date", type=str, default=None, help="Optional expected harvest YYYY-MM-DD")
    p.add_argument(
        "--negatives-ratio",
        type=float,
        default=None,
        help="Add up to this many negatives per positive (e.g. 1.0)",
    )
    p.add_argument("--max-negatives", type=int, default=None, help="Cap negative samples")
    p.add_argument("--association-hours", type=float, default=2.0, help="Spacing for negative sampling")
    p.add_argument(
        "--hybrid-labeling",
        action="store_true",
        help="Enable confidence-aware hybrid labeling filters for positives/negatives.",
    )
    p.add_argument(
        "--hybrid-min-negative-soil-moisture",
        type=float,
        default=50.0,
        help="Hybrid mode: minimum soil moisture for high-confidence negatives.",
    )
    p.add_argument(
        "--hybrid-max-negative-vpd-kpa",
        type=float,
        default=1.2,
        help="Hybrid mode: maximum VPD for high-confidence negatives.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-audit", action="store_true", help="Skip data audit")
    p.add_argument(
        "--audit-profile",
        type=str,
        choices=("field", "synthetic"),
        default="field",
        help="field = safety+bounds for real pulls; synthetic = full checks like generator",
    )
    p.add_argument("--no-dotenv", action="store_true", help="Do not load .env from project root")
    return p.parse_args()


def _parse_iso_date(s: str):
    from datetime import date

    return date.fromisoformat(s)


def _parse_optional_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    if not args.no_dotenv:
        load_dotenv(root / ".env")
        load_dotenv(root / ".env.local")

    base_url = args.base_url or os.environ.get("THINGSPEAK_BASE_URL", "https://api.thingspeak.com")
    env_cid = args.env_channel_id or os.environ.get("THINGSPEAK_ENV_CHANNEL_ID")
    env_key = args.env_read_key or os.environ.get("THINGSPEAK_ENV_READ_API_KEY")
    irr_cid = args.irr_channel_id or os.environ.get("THINGSPEAK_IRRIGATION_CHANNEL_ID")
    irr_key = args.irr_read_key or os.environ.get("THINGSPEAK_IRRIGATION_READ_API_KEY")

    missing = [k for k, v in [
        ("THINGSPEAK_ENV_CHANNEL_ID / --env-channel-id", env_cid),
        ("THINGSPEAK_ENV_READ_API_KEY / --env-read-key", env_key),
        ("THINGSPEAK_IRRIGATION_CHANNEL_ID / --irr-channel-id", irr_cid),
        ("THINGSPEAK_IRRIGATION_READ_API_KEY / --irr-read-key", irr_key),
    ] if not v]
    if missing:
        print("Missing configuration:", ", ".join(missing), file=sys.stderr)
        sys.exit(1)

    range_end = _parse_optional_dt(args.range_end) or datetime.now(UTC)
    if args.range_start:
        range_start = _parse_optional_dt(args.range_start)
        if range_start is None:
            print("Invalid --range-start", file=sys.stderr)
            sys.exit(1)
    else:
        range_start = range_end - timedelta(days=float(args.days))

    if range_end <= range_start:
        print("range_end must be after range_start", file=sys.stderr)
        sys.exit(1)

    planting_date = _parse_iso_date(args.planting_date)
    harvest_date = _parse_iso_date(args.harvest_date) if args.harvest_date else None

    print("Fetching environmental channel…")
    env_res = fetch_feeds_timerange(
        base_url=base_url,
        channel_id=str(env_cid),
        api_key=str(env_key),
        range_start=range_start,
        range_end=range_end,
        chunk_days=float(args.chunk_days),
        pause_seconds=float(args.pause_seconds),
        max_feeds=args.max_feeds_per_channel,
    )
    print(f"  env feeds: {len(env_res.feeds)}  HTTP calls: {env_res.requests_made}")

    print("Fetching irrigation log channel…")
    irr_res = fetch_feeds_timerange(
        base_url=base_url,
        channel_id=str(irr_cid),
        api_key=str(irr_key),
        range_start=range_start,
        range_end=range_end,
        chunk_days=float(args.chunk_days),
        pause_seconds=float(args.pause_seconds),
        max_feeds=args.max_feeds_per_channel,
    )
    print(f"  irrigation feeds: {len(irr_res.feeds)}  HTTP calls: {irr_res.requests_made}")

    records, stats = build_training_records(
        env_res.feeds,
        irr_res.feeds,
        farm_id=args.farm_id,
        field_id=args.field_id,
        node_id=args.node_id,
        planting_date=planting_date,
        expected_harvest_date=harvest_date,
        dataset_version=args.version,
        association_hours=float(args.association_hours),
        max_negatives=args.max_negatives,
        negatives_ratio=args.negatives_ratio,
        random_seed=int(args.seed),
        hybrid_labeling=bool(args.hybrid_labeling),
        hybrid_min_negative_soil_moisture=float(args.hybrid_min_negative_soil_moisture),
        hybrid_max_negative_vpd_kpa=float(args.hybrid_max_negative_vpd_kpa),
    )

    print("── Builder stats ──")
    print(f"  irrigation events (duration>0): {stats.n_irrigation_rows}")
    print(f"  positive training rows:         {stats.n_positive_merged}")
    print(f"  skipped (no env merge):         {stats.n_positive_skipped_no_env}")
    print(f"  negative rows added:            {stats.n_negative_added}")
    print(f"  skipped by hybrid (positive):   {stats.n_positive_skipped_hybrid_rules}")
    print(f"  skipped by hybrid (negative):   {stats.n_negative_skipped_hybrid_rules}")

    if not records:
        print("No records produced — check channels, date range, and field mappings.", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame([r.model_dump() for r in records])
    print("\n── Label report ──")
    if "label_source" in df.columns:
        print(df["label_source"].value_counts(dropna=False).to_string())
    print("\nClass balance:")
    print(df["irrigation_needed"].value_counts(dropna=False).to_string())

    if not args.skip_audit:
        print(f"\nRunning data audit (profile={args.audit_profile})…")
        audit = DataAudit()
        report = audit.run(df, profile=args.audit_profile)
        print(report.summary())
        if not report.passed:
            print(
                "\nAudit failed — fix data, use --audit-profile synthetic only on large field "
                "sets, or --skip-audit (not recommended).",
                file=sys.stderr,
            )
            sys.exit(1)

    out = args.output
    if out is None:
        out = root / "datasets" / "raw" / args.version / "training_records.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    print(f"\nWrote {len(df):,} rows → {out}")


if __name__ == "__main__":
    main()
