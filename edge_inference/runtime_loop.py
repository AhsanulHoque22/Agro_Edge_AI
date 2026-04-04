"""
Runtime decision loop utilities for AgroEdge edge deployment.

Provides:
  - telemetry -> feature payload mapping
  - one-cycle decision execution (dry-run by default)
  - structured action payload generation for later ESP32 integration
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from data_pipeline.schemas import FieldMetadata, encode_growth_stage
from data_pipeline.schemas.feature_record import compute_temp_humidity_index, compute_vpd
from edge_inference.inference_engine import DecisionResult, EdgeInferenceEngine


@dataclass
class RuntimeContext:
    """Context not directly present in ThingSpeak environmental row."""

    field_metadata: FieldMetadata
    days_since_last_irrigation: int


def build_feature_payload(telemetry_row: dict[str, Any], context: RuntimeContext) -> dict[str, Any]:
    """
    Build model feature payload from telemetry + farm metadata context.
    """
    field = context.field_metadata
    stage = field.current_growth_stage()
    dat = field.days_after_transplanting()
    dtm = field.days_to_maturity()

    collected_raw = telemetry_row.get("collected_at")
    if isinstance(collected_raw, str):
        collected_dt = datetime.fromisoformat(collected_raw.replace("Z", "+00:00"))
    else:
        collected_dt = datetime.now(UTC)

    air_temp = float(telemetry_row["air_temperature_celsius"])
    air_hum = float(telemetry_row["air_humidity_percent"])
    vpd = compute_vpd(air_temp, air_hum)
    thi = compute_temp_humidity_index(air_temp, air_hum)
    soil_now = float(telemetry_row["soil_moisture_percent"])
    soil_t1 = float(telemetry_row.get("soil_moisture_percent_t_minus_1", soil_now))
    soil_t2 = float(telemetry_row.get("soil_moisture_percent_t_minus_2", soil_t1))
    delta_t1 = float(telemetry_row.get("delta_soil_moisture_percent_t_minus_1", soil_now - soil_t1))
    delta_t2 = float(telemetry_row.get("delta_soil_moisture_percent_t_minus_2", soil_now - soil_t2))
    time_since_last_hours = float(
        telemetry_row.get(
            "time_since_last_irrigation_hours",
            float(context.days_since_last_irrigation) * 24.0,
        )
    )
    if "days_since_last_irrigation" in telemetry_row:
        dsirr = int(telemetry_row["days_since_last_irrigation"])
    else:
        dsirr = int(max(0.0, time_since_last_hours) // 24)
    last_irrigation_duration_minutes = float(telemetry_row.get("last_irrigation_duration_minutes", 0.0))
    rain_expected_flag = int(telemetry_row.get("rain_expected_flag", 0))

    payload = {
        "soil_moisture_percent": soil_now,
        "soil_temperature_celsius": float(telemetry_row["soil_temperature_celsius"]),
        "air_temperature_celsius": air_temp,
        "air_humidity_percent": air_hum,
        "light_intensity_lux": float(telemetry_row["light_intensity_lux"]),
        "water_tank_level_percent": float(telemetry_row["water_tank_level_percent"]),
        "growth_stage_encoded": int(encode_growth_stage(stage)),
        "days_after_transplanting": int(dat),
        "days_to_maturity": int(dtm),
        "vpd_kpa": float(vpd),
        "hour_of_day": int(collected_dt.hour),
        "day_of_week": int(collected_dt.weekday()),
        "month": int(collected_dt.month),
        "days_since_last_irrigation": dsirr,
        "soil_moisture_percent_t_minus_1": soil_t1,
        "soil_moisture_percent_t_minus_2": soil_t2,
        "delta_soil_moisture_percent_t_minus_1": delta_t1,
        "delta_soil_moisture_percent_t_minus_2": delta_t2,
        "time_since_last_irrigation_hours": time_since_last_hours,
        "last_irrigation_duration_minutes": last_irrigation_duration_minutes,
        "rain_expected_flag": rain_expected_flag,
        "temp_humidity_index": thi,
    }
    return payload


def build_action_payload(decision: DecisionResult, telemetry_row: dict[str, Any], node_id: str) -> dict[str, Any]:
    """
    Build an action/log payload for future command publishing + irrigation logs.
    """
    trigger_source = "ai_model"
    return {
        "action_timestamp_utc": datetime.now(UTC).isoformat(),
        "node_id": node_id,
        "should_irrigate": decision.should_irrigate,
        "approved_duration_minutes": round(decision.approved_duration_minutes, 2),
        "blocked_reason": decision.blocked_reason,
        "trigger_source": trigger_source,
        "soil_moisture_before_percent": float(telemetry_row["soil_moisture_percent"]),
        "model_probability": round(decision.model_prediction.irrigation_probability, 4),
        "model_predicted_duration": round(decision.model_prediction.irrigation_duration_minutes, 2),
        "model_version": decision.model_version,
    }


def run_one_cycle(
    engine: EdgeInferenceEngine,
    telemetry_row: dict[str, Any],
    context: RuntimeContext,
    node_id: str,
) -> tuple[dict[str, Any], DecisionResult, dict[str, Any]]:
    """
    Execute one runtime cycle:
      telemetry -> features -> decision -> action payload
    """
    feature_payload = build_feature_payload(telemetry_row, context)
    decision = engine.decide(feature_payload)
    action_payload = build_action_payload(decision, telemetry_row, node_id=node_id)
    return feature_payload, decision, action_payload
