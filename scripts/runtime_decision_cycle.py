"""
Run one AgroEdge runtime decision cycle (dry-run).

Modes:
  - live ThingSpeak fetch (requires .env configured) — uses the same enrichment
    as ``runtime_scheduler.py`` (latest-N env rows, lags/deltas, irrigation context).
  - sample telemetry fallback (default, no network needed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloud_integration.thingspeak_client import (
    ThingSpeakConfig,
    ThingSpeakReadClient,
    ThingSpeakWriteClient,
)
from cloud_integration.thingspeak_client.enriched_telemetry import (
    EnrichedFetchState,
    attach_sample_monitoring,
    fetch_enriched_environment_row,
)
from data_pipeline.schemas import FieldMetadata
from edge_inference import EdgeInferenceEngine, RuntimeContext, run_one_cycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one edge runtime decision cycle.")
    parser.add_argument("--bundle-version", type=str, default="v1.0.0")
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use built-in sample telemetry instead of ThingSpeak.",
    )
    parser.add_argument(
        "--env-history-rows",
        type=int,
        default=3,
        help="ThingSpeak env rows to pull for lag features (live mode; default: 3).",
    )
    parser.add_argument("--days-since-last-irrigation", type=int, default=2)
    parser.add_argument("--planting-date", type=str, default="2025-01-01", help="YYYY-MM-DD")
    parser.add_argument(
        "--publish-log",
        action="store_true",
        help="Publish action payload to ThingSpeak irrigation log channel.",
    )
    return parser.parse_args()


def sample_telemetry() -> dict[str, float | str | None]:
    return {
        "collected_at": "2026-03-19T08:15:00+00:00",
        "soil_moisture_percent": 46.2,
        "soil_temperature_celsius": 27.4,
        "air_temperature_celsius": 31.6,
        "air_humidity_percent": 72.0,
        "light_intensity_lux": 42000.0,
        "water_tank_level_percent": 78.0,
        "water_flow_rate_lph": None,
        "entry_id": -1,
    }


def build_read_client_from_env() -> ThingSpeakReadClient:
    """Same read client as runtime_scheduler (env + irrigation context channels)."""
    cfg = ThingSpeakConfig(
        base_url=os.getenv("THINGSPEAK_BASE_URL", "https://api.thingspeak.com"),
        env_channel_id=os.environ["THINGSPEAK_ENV_CHANNEL_ID"],
        env_read_api_key=os.environ["THINGSPEAK_ENV_READ_API_KEY"],
        irrigation_channel_id=os.getenv("THINGSPEAK_IRRIGATION_CHANNEL_ID"),
        irrigation_read_api_key=os.getenv("THINGSPEAK_IRRIGATION_READ_API_KEY"),
    )
    return ThingSpeakReadClient(cfg)


def fetch_live_enriched_telemetry(env_history_rows: int) -> dict[str, Any]:
    load_dotenv()
    client = build_read_client_from_env()
    state = EnrichedFetchState()
    return fetch_enriched_environment_row(
        client, state, env_history_rows=max(1, int(env_history_rows))
    )


def build_write_client_from_env() -> ThingSpeakWriteClient:
    load_dotenv()
    cfg = ThingSpeakConfig(
        base_url=os.getenv("THINGSPEAK_BASE_URL", "https://api.thingspeak.com"),
        env_channel_id=os.getenv("THINGSPEAK_ENV_CHANNEL_ID", ""),
        env_read_api_key=os.getenv("THINGSPEAK_ENV_READ_API_KEY", ""),
        irrigation_channel_id=os.environ["THINGSPEAK_IRRIGATION_CHANNEL_ID"],
        irrigation_write_api_key=os.environ["THINGSPEAK_IRRIGATION_WRITE_API_KEY"],
    )
    return ThingSpeakWriteClient(cfg)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    bundle_dir = root / "model_export" / "irrigation_model" / args.bundle_version
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_dir}")

    if args.use_sample:
        telemetry = attach_sample_monitoring(sample_telemetry())
    else:
        telemetry = fetch_live_enriched_telemetry(args.env_history_rows)

    field_metadata = FieldMetadata(
        farm_id=os.getenv("DEFAULT_FARM_ID", "farm_001"),
        field_name=os.getenv("DEFAULT_FIELD_ID", "field_001"),
        node_id=os.getenv("DEFAULT_NODE_ID", "node_esp32_001"),
        planting_date=date.fromisoformat(args.planting_date),
    )
    field_metadata.update_growth_stage()

    dsirr = args.days_since_last_irrigation
    if not args.use_sample and "days_since_last_irrigation" in telemetry:
        dsirr = int(telemetry["days_since_last_irrigation"])

    context = RuntimeContext(
        field_metadata=field_metadata,
        days_since_last_irrigation=dsirr,
    )
    engine = EdgeInferenceEngine(bundle_dir=bundle_dir)
    features, decision, action = run_one_cycle(
        engine=engine,
        telemetry_row=telemetry,
        context=context,
        node_id=field_metadata.node_id,
    )

    print("Runtime Decision Cycle (dry-run)")
    mon = telemetry.pop("_runtime_monitoring", None)
    print("telemetry:")
    print(json.dumps(telemetry, indent=2))
    if mon is not None:
        print("\n_runtime_monitoring:")
        print(json.dumps(mon, indent=2))
    print("\nfeature_payload:")
    print(json.dumps(features, indent=2))
    print("\ndecision:")
    print(
        json.dumps(
            {
                "should_irrigate": decision.should_irrigate,
                "approved_duration_minutes": decision.approved_duration_minutes,
                "blocked_reason": decision.blocked_reason,
                "model_probability": decision.model_prediction.irrigation_probability,
                "model_predicted_duration": decision.model_prediction.irrigation_duration_minutes,
                "model_version": decision.model_version,
            },
            indent=2,
        )
    )
    print("\naction_payload:")
    print(json.dumps(action, indent=2))

    if args.publish_log:
        writer = build_write_client_from_env()
        entry_id = writer.publish_irrigation_log(action)
        print(f"\nIrrigation log published to ThingSpeak with entry_id={entry_id}")
    else:
        print("\nLog publish skipped (dry-run). Use --publish-log to send to ThingSpeak.")


if __name__ == "__main__":
    main()
