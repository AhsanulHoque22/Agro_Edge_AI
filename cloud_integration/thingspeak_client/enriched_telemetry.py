"""
Shared ThingSpeak → telemetry row enrichment for runtime inference.

Used by ``scripts/runtime_scheduler.py`` and ``scripts/runtime_decision_cycle.py``
so single-cycle debugging matches production: latest-N env rows, ring-buffer
lags/deltas, irrigation log context, and structured ``_runtime_monitoring``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .client import ThingSpeakReadClient


@dataclass
class EnrichedFetchState:
    """Mutable state for repeated fetches (scheduler) or one-shot with defaults (decision cycle)."""

    moisture_ring: deque[float] = field(default_factory=lambda: deque(maxlen=3))
    last_good_telemetry: dict[str, Any] | None = None
    last_irrigation_context: dict[str, Any] | None = None


def default_sample_runtime_monitoring() -> dict[str, Any]:
    """Monitoring block for ``--use-sample`` / offline rows."""
    return {
        "degraded_mode": False,
        "degraded_reasons": [],
        "source_mode": "sample",
        "env_history_rows_target": 1,
        "env_history_rows_available": 1,
        "used_cached_telemetry": False,
        "used_cached_irrigation_context": False,
        "rain_expected_flag_source": "default_zero",
    }


def attach_sample_monitoring(telemetry_row: dict[str, Any]) -> dict[str, Any]:
    """Copy row and attach sample-mode monitoring (matches scheduler sample path)."""
    out = dict(telemetry_row)
    out["_runtime_monitoring"] = default_sample_runtime_monitoring()
    return out


def fetch_enriched_environment_row(
    client: ThingSpeakReadClient,
    state: EnrichedFetchState,
    *,
    env_history_rows: int = 3,
) -> dict[str, Any]:
    """
    Fetch latest environment history + irrigation context; merge v1.1 feature hints.

    Updates ``state`` (ring buffer, caches). On first env failure with no cache, re-raises.
    """
    env_history_target = max(1, int(env_history_rows))
    monitoring: dict[str, Any] = {
        "degraded_mode": False,
        "degraded_reasons": [],
        "source_mode": "thingspeak",
        "env_history_rows_target": env_history_target,
        "env_history_rows_available": 0,
        "used_cached_telemetry": False,
        "used_cached_irrigation_context": False,
        "rain_expected_flag_source": "default_zero",
    }

    try:
        history_rows = client.fetch_latest_environment_rows(results=env_history_target)
        monitoring["env_history_rows_available"] = len(history_rows)
        current = dict(history_rows[-1])
        for item in history_rows:
            state.moisture_ring.append(float(item["soil_moisture_percent"]))
    except Exception as exc:  # noqa: BLE001
        monitoring["degraded_mode"] = True
        monitoring["degraded_reasons"].append(f"env_cloud_fetch_failed:{type(exc).__name__}")
        if state.last_good_telemetry is None:
            raise
        current = dict(state.last_good_telemetry)
        monitoring["used_cached_telemetry"] = True

    irr: dict[str, Any] | None
    cfg = client.config
    if cfg.irrigation_channel_id and cfg.irrigation_read_api_key:
        try:
            irr = client.fetch_latest_irrigation_event_row()
            state.last_irrigation_context = irr
        except Exception as exc:  # noqa: BLE001
            monitoring["degraded_mode"] = True
            monitoring["degraded_reasons"].append(
                f"irrigation_context_fetch_failed:{type(exc).__name__}"
            )
            irr = state.last_irrigation_context
            monitoring["used_cached_irrigation_context"] = irr is not None
    else:
        irr = state.last_irrigation_context
        monitoring["irrigation_context_skipped"] = "channel_or_read_key_not_configured"

    if irr is not None:
        current["last_irrigation_duration_minutes"] = float(
            irr.get("irrigation_duration_minutes", 0.0)
        )
        try:
            current_ts = datetime.fromisoformat(
                str(current.get("collected_at", "")).replace("Z", "+00:00")
            )
            if current_ts.tzinfo is None:
                current_ts = current_ts.replace(tzinfo=UTC)
            irr_ts = datetime.fromisoformat(str(irr.get("created_at", "")).replace("Z", "+00:00"))
            if irr_ts.tzinfo is None:
                irr_ts = irr_ts.replace(tzinfo=UTC)
            hours = max(
                0.0,
                (current_ts.astimezone(UTC) - irr_ts.astimezone(UTC)).total_seconds() / 3600.0,
            )
            current["time_since_last_irrigation_hours"] = round(hours, 2)
            current["days_since_last_irrigation"] = int(hours // 24)
        except Exception:  # noqa: BLE001
            pass

    soil_now = float(current["soil_moisture_percent"])
    if len(state.moisture_ring) >= 2:
        soil_t1 = float(state.moisture_ring[-2])
    else:
        soil_t1 = soil_now
    if len(state.moisture_ring) >= 3:
        soil_t2 = float(state.moisture_ring[-3])
    else:
        soil_t2 = soil_t1
    current["soil_moisture_percent_t_minus_1"] = soil_t1
    current["soil_moisture_percent_t_minus_2"] = soil_t2
    current["delta_soil_moisture_percent_t_minus_1"] = round(soil_now - soil_t1, 3)
    current["delta_soil_moisture_percent_t_minus_2"] = round(soil_now - soil_t2, 3)
    current["rain_expected_flag"] = 0
    current["_runtime_monitoring"] = monitoring
    current["moisture_ring_size"] = len(state.moisture_ring)
    state.last_good_telemetry = dict(current)
    return current
