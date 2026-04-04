"""ThingSpeak feed → TrainingRecord builder."""

from __future__ import annotations

from datetime import UTC, date, datetime

from data_pipeline.ingestion.thingspeak_training_builder import build_training_records
from data_pipeline.schemas import LabelSource


def _env(entry_id: int, ts: str, soil: float = 42.0) -> dict:
    return {
        "entry_id": entry_id,
        "created_at": ts,
        "field1": str(soil),
        "field2": "28.0",
        "field3": "31.0",
        "field4": "65.0",
        "field5": "25000.0",
        "field6": "80.0",
        "field7": "",
    }


def _irr(entry_id: int, ts: str, minutes: float = 15.0) -> dict:
    return {
        "entry_id": entry_id,
        "created_at": ts,
        "field1": str(minutes),
        "field2": "0",
        "field3": str(42.0),
        "field4": "",
        "field5": "123",
    }


def test_build_training_records_positive_merge() -> None:
    env_feeds = [
        _env(1, "2024-07-10T10:00:00Z", soil=40.0),
        _env(2, "2024-07-10T10:14:59Z", soil=39.0),
    ]
    irr_feeds = [_irr(10, "2024-07-10T10:15:00Z", minutes=20.0)]

    records, stats = build_training_records(
        env_feeds,
        irr_feeds,
        farm_id="f1",
        field_id="fd1",
        node_id="n1",
        planting_date=date(2024, 5, 1),
        expected_harvest_date=None,
        dataset_version="t1",
    )

    assert stats.n_positive_merged == 1
    assert stats.n_irrigation_rows == 1
    assert len(records) == 1
    rec = records[0]
    assert rec.irrigation_needed == 1
    assert rec.irrigation_duration_minutes == 20.0
    assert rec.label_source == LabelSource.THINGSPEAK_HISTORICAL
    assert rec.soil_moisture_percent == 39.0  # last env at or before irr time
    assert rec.collected_at == datetime(2024, 7, 10, 10, 15, tzinfo=UTC)
    assert rec.soil_moisture_percent_t_minus_1 == 40.0
    assert rec.soil_moisture_percent_t_minus_2 == 40.0
    assert rec.delta_soil_moisture_percent_t_minus_1 == -1.0
    assert rec.time_since_last_irrigation_hours >= 0.0
    assert rec.rain_expected_flag == 0


def test_build_training_records_with_negatives() -> None:
    env_feeds = [
        _env(1, "2024-07-10T08:00:00Z"),
        _env(2, "2024-07-10T20:00:00Z"),
    ]
    irr_feeds = [_irr(10, "2024-07-10T10:00:00Z", minutes=10.0)]

    records, stats = build_training_records(
        env_feeds,
        irr_feeds,
        farm_id="f1",
        field_id="fd1",
        node_id="n1",
        planting_date=date(2024, 5, 1),
        expected_harvest_date=None,
        dataset_version="t1",
        negatives_ratio=2.0,
        max_negatives=5,
        association_hours=2.0,
        random_seed=0,
    )

    assert stats.n_positive_merged == 1
    assert stats.n_negative_added >= 1
    assert sum(1 for r in records if r.irrigation_needed == 0) >= 1


def test_build_training_records_hybrid_labeling_filters_low_confidence_negatives() -> None:
    env_feeds = [
        # Candidate negative with low moisture (filtered in hybrid mode)
        _env(1, "2024-07-10T20:00:00Z", soil=38.0),
        # Candidate negative with high moisture (kept in hybrid mode)
        _env(2, "2024-07-10T21:00:00Z", soil=62.0),
    ]
    irr_feeds = [_irr(10, "2024-07-10T10:00:00Z", minutes=12.0)]

    records, stats = build_training_records(
        env_feeds,
        irr_feeds,
        farm_id="f1",
        field_id="fd1",
        node_id="n1",
        planting_date=date(2024, 5, 1),
        expected_harvest_date=None,
        dataset_version="t1",
        negatives_ratio=3.0,
        max_negatives=10,
        association_hours=2.0,
        random_seed=0,
        hybrid_labeling=True,
        hybrid_min_negative_soil_moisture=50.0,
        hybrid_max_negative_vpd_kpa=2.0,
    )

    negatives = [r for r in records if r.irrigation_needed == 0]
    assert stats.n_negative_skipped_hybrid_rules >= 1
    assert len(negatives) >= 1
    assert all(r.label_source == LabelSource.RULE_DERIVED for r in negatives)
