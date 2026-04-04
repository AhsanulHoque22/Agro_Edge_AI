"""
FeatureRecord — the feature vector presented to the irrigation model.

This is the strict contract between the data pipeline and the ML model.
Derived from SensorRecord + FieldMetadata + irrigation log data.

WARNING: The field ORDER in to_model_input() is fixed after training.
         It must match the feature list in configs/feature_schema.yaml.
         Do not reorder without retraining and re-exporting the model.
"""

from __future__ import annotations

import math
from datetime import datetime

from pydantic import BaseModel, Field


def compute_vpd(air_temperature_celsius: float, air_humidity_percent: float) -> float:
    """
    Compute Vapor Pressure Deficit (kPa).

    VPD quantifies the drying power of the atmosphere. High VPD means the air
    can absorb much more moisture — plants transpire faster and need more water.
    It is derived entirely from existing sensor readings, requiring no extra hardware.

    Formula (Tetens approximation):
        SVP = 0.6108 * exp(17.27 * T / (T + 237.3))
        VPD = (1 - RH/100) * SVP

    Args:
        air_temperature_celsius: Ambient air temperature (°C).
        air_humidity_percent:    Relative humidity (%).

    Returns:
        VPD in kilopascals (kPa), rounded to 4 decimal places.
    """
    saturation_vapor_pressure = 0.6108 * math.exp(
        17.27 * air_temperature_celsius / (air_temperature_celsius + 237.3)
    )
    return round((1.0 - air_humidity_percent / 100.0) * saturation_vapor_pressure, 4)


def compute_temp_humidity_index(air_temperature_celsius: float, air_humidity_percent: float) -> float:
    """
    Temperature–humidity index (THI-style comfort/stress scalar) from air T and RH.

    Same formula as docs/MODEL_INTELLIGENCE_V1_1.md and runtime inference.
    """
    thi = air_temperature_celsius - (
        (0.55 - 0.0055 * air_humidity_percent) * (air_temperature_celsius - 14.5)
    )
    return float(round(thi, 4))


class FeatureRecord(BaseModel):
    """
    Feature vector for irrigation model inference.

    14 features total. The identity fields (record_id, farm_id, field_id,
    node_id, collected_at) are included for traceability but are NOT passed
    to the model — only to_model_input() produces the model-facing array.
    """

    # ── Identity (traceability only — not model features) ─────────────────────
    record_id: str
    farm_id: str
    field_id: str
    node_id: str
    collected_at: datetime

    # ── Sensor features ───────────────────────────────────────────────────────
    soil_moisture_percent: float = Field(ge=0.0, le=100.0)
    soil_temperature_celsius: float = Field(ge=-10.0, le=80.0)
    air_temperature_celsius: float = Field(ge=-10.0, le=60.0)
    air_humidity_percent: float = Field(ge=0.0, le=100.0)
    light_intensity_lux: float = Field(ge=0.0)
    water_tank_level_percent: float = Field(ge=0.0, le=100.0)

    # ── Crop context features (from FieldMetadata) ────────────────────────────
    growth_stage_encoded: int = Field(ge=0, le=7)
    days_after_transplanting: int = Field(ge=0)
    days_to_maturity: int   # negative values mean harvest is overdue

    # ── Derived features ──────────────────────────────────────────────────────
    vpd_kpa: float = Field(ge=0.0, description="Vapor Pressure Deficit (kPa).")
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    days_since_last_irrigation: int = Field(ge=0)

    # ── v1.1 additive features (optional for backward compatibility) ──────────
    soil_moisture_percent_t_minus_1: float = Field(default=0.0)
    soil_moisture_percent_t_minus_2: float = Field(default=0.0)
    delta_soil_moisture_percent_t_minus_1: float = Field(default=0.0)
    delta_soil_moisture_percent_t_minus_2: float = Field(default=0.0)
    time_since_last_irrigation_hours: float = Field(default=0.0, ge=0.0)
    last_irrigation_duration_minutes: float = Field(default=0.0, ge=0.0)
    rain_expected_flag: int = Field(default=0, ge=0, le=1)
    temp_humidity_index: float = Field(default=0.0)

    schema_version: str = Field(default="1.0")

    def to_model_input(self) -> list[float]:
        """
        Return an ordered list of feature values for model inference.

        Order matches configs/feature_schema.yaml. All values are cast to float
        for compatibility with scikit-learn and numpy.
        """
        return [
            self.soil_moisture_percent,
            self.soil_temperature_celsius,
            self.air_temperature_celsius,
            self.air_humidity_percent,
            self.light_intensity_lux,
            self.water_tank_level_percent,
            float(self.growth_stage_encoded),
            float(self.days_after_transplanting),
            float(self.days_to_maturity),
            self.vpd_kpa,
            float(self.hour_of_day),
            float(self.day_of_week),
            float(self.month),
            float(self.days_since_last_irrigation),
        ]
