"""
SensorRecord — canonical raw telemetry record from an ESP32 node.

This is the first record type in the pipeline. It represents a single sensor
reading cycle before any feature engineering has been applied.

All downstream records (FeatureRecord, TrainingRecord) are derived from this.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SensorRecord(BaseModel):
    """
    Raw environmental reading from an ESP32 sensor node.

    Populated at ingestion time on the Raspberry Pi after retrieving data
    from ThingSpeak. Validated against configs/sensor_schema.yaml.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this reading. Generated at ingestion.",
    )
    farm_id: str = Field(description="Identifies the farm.")
    field_id: str = Field(description="Identifies the crop plot within the farm.")
    node_id: str = Field(description="ESP32 device identifier.")
    collected_at: datetime = Field(description="UTC timestamp when sensor was read.")
    ingested_at: datetime = Field(description="UTC timestamp when Pi received the record.")

    # ── Core sensors (required) ───────────────────────────────────────────────

    soil_moisture_percent: float = Field(
        ge=0.0, le=100.0,
        description="Volumetric soil moisture content at root zone depth (%).",
    )
    soil_temperature_celsius: float = Field(
        ge=-10.0, le=80.0,
        description="Soil temperature at root zone depth (°C). DS18B20 sensor.",
    )
    air_temperature_celsius: float = Field(
        ge=-10.0, le=60.0,
        description="Ambient air temperature (°C). DHT22 sensor.",
    )
    air_humidity_percent: float = Field(
        ge=0.0, le=100.0,
        description="Relative ambient humidity (%). DHT22 sensor.",
    )
    light_intensity_lux: float = Field(
        ge=0.0, le=150000.0,
        description="Photosynthetically active light intensity (lux). BH1750 sensor.",
    )
    water_tank_level_percent: float = Field(
        ge=0.0, le=100.0,
        description="Irrigation water tank fill level (%). Safety check before actuation.",
    )

    # ── Optional sensors (nullable until hardware is installed) ───────────────

    water_flow_rate_lph: Optional[float] = Field(
        default=None, ge=0.0,
        description="Water flow rate through the irrigation pipe (L/hr). Flow sensor.",
    )
    soil_ph: Optional[float] = Field(
        default=None, ge=0.0, le=14.0,
        description="Soil acidity/alkalinity (pH). Future sensor.",
    )
    rainfall_mm: Optional[float] = Field(
        default=None, ge=0.0,
        description="Rainfall accumulation since last reading (mm). Future sensor.",
    )
    leaf_wetness_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Leaf surface wetness (%). Disease risk indicator. Future sensor.",
    )

    schema_version: str = Field(default="1.0")
