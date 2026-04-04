"""Audit profiles (synthetic vs field)."""

from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.validation.data_audit import DataAudit


def test_field_audit_passes_small_realistic_frame() -> None:
    df = pd.DataFrame(
        {
            "irrigation_needed": [1, 0, 1, 0, 1],
            "water_tank_level_percent": [80.0, 80.0, 75.0, 82.0, 78.0],
            "soil_moisture_percent": [30.0, 55.0, 28.0, 60.0, 25.0],
            "vpd_kpa": [1.2, 0.8, 1.3, 0.7, 1.4],
            "growth_stage_encoded": [2, 2, 2, 2, 2],
            "irrigation_duration_minutes": [15.0, 0.0, 20.0, 0.0, 18.0],
            "days_after_transplanting": [40, 40, 40, 40, 40],
            "air_humidity_percent": [60.0, 70.0, 58.0, 72.0, 57.0],
            "days_since_last_irrigation": [2, 3, 1, 4, 1],
            "hour_of_day": [10, 11, 12, 13, 14],
            "month": [7, 7, 7, 7, 7],
        }
    )
    report = DataAudit().run(df, profile="field")
    assert report.passed


def test_field_audit_fails_tiny_dataset() -> None:
    df = pd.DataFrame(
        {
            "irrigation_needed": [1],
            "water_tank_level_percent": [80.0],
            "soil_moisture_percent": [30.0],
            "vpd_kpa": [1.2],
            "growth_stage_encoded": [2],
            "irrigation_duration_minutes": [15.0],
            "days_after_transplanting": [40],
            "air_humidity_percent": [60.0],
            "days_since_last_irrigation": [2],
            "hour_of_day": [10],
            "month": [7],
        }
    )
    report = DataAudit().run(df, profile="field")
    assert not report.passed
    assert any(r.check_name.startswith("Minimum") for r in report.results if not r.passed)


def test_unknown_audit_profile_raises() -> None:
    df = pd.DataFrame({"irrigation_needed": [1]})
    with pytest.raises(ValueError, match="Unknown audit profile"):
        DataAudit().run(df, profile="nope")  # type: ignore[arg-type]
