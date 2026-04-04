"""
Continuous AgroEdge runtime scheduler.

Examples:
  # Bounded local test (no network writes)
  python scripts/runtime_scheduler.py --use-sample --max-cycles 3 --interval-seconds 2

  # Continuous live loop with ThingSpeak log publishing
  python scripts/runtime_scheduler.py --publish-log
"""

from __future__ import annotations

import argparse
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
from edge_inference import (
    EdgeInferenceEngine,
    RetryPolicy,
    RuntimeContext,
    RuntimeScheduler,
    SchedulerConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous AgroEdge runtime scheduler.")
    parser.add_argument("--bundle-version", type=str, default="v1.0.0")
    parser.add_argument("--use-sample", action="store_true", help="Use built-in sample telemetry.")
    parser.add_argument(
        "--publish-log",
        action="store_true",
        help="Publish action log to ThingSpeak irrigation channel.",
    )
    parser.add_argument("--interval-seconds", type=float, default=900.0)
    parser.add_argument("--max-cycles", type=int, default=None, help="Stop after N cycles (for testing).")
    parser.add_argument("--days-since-last-irrigation", type=int, default=2)
    parser.add_argument("--planting-date", type=str, default="2025-01-01", help="YYYY-MM-DD")
    parser.add_argument("--retry-max-attempts", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--retry-max-delay", type=float, default=8.0)
    parser.add_argument("--env-history-rows", type=int, default=3)
    parser.add_argument("--log-path", type=str, default="logs/runtime_cycles.jsonl")
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
    cfg = ThingSpeakConfig(
        base_url=os.getenv("THINGSPEAK_BASE_URL", "https://api.thingspeak.com"),
        env_channel_id=os.environ["THINGSPEAK_ENV_CHANNEL_ID"],
        env_read_api_key=os.environ["THINGSPEAK_ENV_READ_API_KEY"],
        irrigation_channel_id=os.getenv("THINGSPEAK_IRRIGATION_CHANNEL_ID"),
        irrigation_read_api_key=os.getenv("THINGSPEAK_IRRIGATION_READ_API_KEY"),
    )
    return ThingSpeakReadClient(cfg)


def build_write_client_from_env() -> ThingSpeakWriteClient:
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
    load_dotenv()
    root = Path(__file__).resolve().parent.parent
    bundle_dir = root / "model_export" / "irrigation_model" / args.bundle_version
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_dir}")

    field_metadata = FieldMetadata(
        farm_id=os.getenv("DEFAULT_FARM_ID", "farm_001"),
        field_name=os.getenv("DEFAULT_FIELD_ID", "field_001"),
        node_id=os.getenv("DEFAULT_NODE_ID", "node_esp32_001"),
        planting_date=date.fromisoformat(args.planting_date),
    )
    field_metadata.update_growth_stage()
    context = RuntimeContext(
        field_metadata=field_metadata,
        days_since_last_irrigation=args.days_since_last_irrigation,
    )

    engine = EdgeInferenceEngine(bundle_dir=bundle_dir)

    if args.use_sample:
        def fetch_fn() -> dict[str, Any]:
            return attach_sample_monitoring(sample_telemetry())
    else:
        read_client = build_read_client_from_env()
        env_history_target = max(1, int(args.env_history_rows))
        enrich_state = EnrichedFetchState()

        def fetch_fn() -> dict[str, Any]:
            return fetch_enriched_environment_row(
                read_client,
                enrich_state,
                env_history_rows=env_history_target,
            )

    publish_fn = None
    if args.publish_log:
        write_client = build_write_client_from_env()

        def publish_fn(action_payload: dict[str, Any]) -> int:  # type: ignore[no-redef]
            return write_client.publish_irrigation_log(action_payload)

    scheduler = RuntimeScheduler(
        config=SchedulerConfig(
            interval_seconds=args.interval_seconds,
            publish_log=args.publish_log,
            retry_policy=RetryPolicy(
                max_attempts=args.retry_max_attempts,
                base_delay_seconds=args.retry_base_delay,
                max_delay_seconds=args.retry_max_delay,
            ),
            max_cycles=args.max_cycles,
        ),
        runtime_context=context,
        node_id=field_metadata.node_id,
        engine=engine,
        fetch_telemetry_fn=fetch_fn,
        publish_log_fn=publish_fn,
        local_log_path=root / args.log_path,
    )
    scheduler.run()
    print(f"Scheduler stopped. Local log: {root / args.log_path}")


if __name__ == "__main__":
    main()
