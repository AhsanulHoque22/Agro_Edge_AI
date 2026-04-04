"""merge_training_parquet script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_merge_two_parquets(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "merge_training_parquet.py"
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    pd.DataFrame(
        {
            "record_id": ["x"],
            "irrigation_needed": [1],
            "irrigation_duration_minutes": [10.0],
            "dataset_version": ["a"],
            "soil_moisture_percent": [40.0],
            "soil_temperature_celsius": [28.0],
            "air_temperature_celsius": [30.0],
            "air_humidity_percent": [60.0],
            "light_intensity_lux": [1000.0],
            "water_tank_level_percent": [80.0],
            "growth_stage_encoded": [2],
            "days_after_transplanting": [30],
            "days_to_maturity": [100],
            "vpd_kpa": [1.0],
            "hour_of_day": [12],
            "day_of_week": [0],
            "month": [6],
            "days_since_last_irrigation": [1],
            "farm_id": ["f"],
            "field_id": ["fd"],
            "node_id": ["n"],
            "collected_at": [pd.Timestamp("2024-01-01T00:00:00Z")],
            "label_source": ["synthetic"],
            "schema_version": ["1.0"],
        }
    ).to_parquet(a, index=False)
    pd.DataFrame(
        {
            "record_id": ["y"],
            "irrigation_needed": [0],
            "irrigation_duration_minutes": [0.0],
            "dataset_version": ["b"],
            "soil_moisture_percent": [50.0],
            "soil_temperature_celsius": [28.0],
            "air_temperature_celsius": [30.0],
            "air_humidity_percent": [60.0],
            "light_intensity_lux": [1000.0],
            "water_tank_level_percent": [80.0],
            "growth_stage_encoded": [2],
            "days_after_transplanting": [30],
            "days_to_maturity": [100],
            "vpd_kpa": [1.0],
            "hour_of_day": [12],
            "day_of_week": [0],
            "month": [6],
            "days_since_last_irrigation": [2],
            "farm_id": ["f"],
            "field_id": ["fd"],
            "node_id": ["n"],
            "collected_at": [pd.Timestamp("2024-01-02T00:00:00Z")],
            "label_source": ["synthetic"],
            "schema_version": ["1.0"],
        }
    ).to_parquet(b, index=False)
    out = tmp_path / "merged.parquet"
    r = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(a),
            "--input",
            str(b),
            "--output",
            str(out),
            "--dataset-version",
            "m1",
            "--seed",
            "0",
        ],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Wrote 2 rows" in r.stdout
    df = pd.read_parquet(out, engine="pyarrow")
    assert len(df) == 2
    assert set(df["dataset_version"]) == {"m1"}
    assert df["record_id"].nunique() == 2
