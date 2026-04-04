"""Tests for ThingSpeak runtime telemetry enrichment (scheduler / decision cycle)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cloud_integration.thingspeak_client.enriched_telemetry import (
    EnrichedFetchState,
    attach_sample_monitoring,
    fetch_enriched_environment_row,
)


def test_fetch_enriched_merges_lags_and_irrigation() -> None:
    client = MagicMock()
    # Client returns oldest → newest (see ThingSpeakReadClient.fetch_latest_environment_rows).
    client.fetch_latest_environment_rows.return_value = [
        {"soil_moisture_percent": 42.0, "collected_at": "2025-06-01T08:00:00+00:00"},
        {"soil_moisture_percent": 40.0, "collected_at": "2025-06-01T09:00:00+00:00"},
        {"soil_moisture_percent": 38.0, "collected_at": "2025-06-01T10:00:00+00:00"},
    ]
    client.fetch_latest_irrigation_event_row.return_value = {
        "created_at": "2025-06-01T04:00:00+00:00",
        "irrigation_duration_minutes": 35.5,
    }
    state = EnrichedFetchState()
    row = fetch_enriched_environment_row(client, state, env_history_rows=3)
    assert row["soil_moisture_percent"] == 38.0
    assert row["soil_moisture_percent_t_minus_1"] == 40.0
    assert row["soil_moisture_percent_t_minus_2"] == 42.0
    assert row["delta_soil_moisture_percent_t_minus_1"] == pytest.approx(-2.0, rel=1e-3)
    assert row["delta_soil_moisture_percent_t_minus_2"] == pytest.approx(-4.0, rel=1e-3)
    assert row["last_irrigation_duration_minutes"] == 35.5
    assert row["time_since_last_irrigation_hours"] == pytest.approx(6.0, rel=1e-2)
    assert row["days_since_last_irrigation"] == 0
    assert row["_runtime_monitoring"]["source_mode"] == "thingspeak"
    assert row["_runtime_monitoring"]["degraded_mode"] is False


def test_attach_sample_monitoring() -> None:
    row = attach_sample_monitoring({"soil_moisture_percent": 50.0})
    assert row["soil_moisture_percent"] == 50.0
    assert row["_runtime_monitoring"]["source_mode"] == "sample"


def test_enrichment_skips_irrigation_when_not_configured() -> None:
    client = MagicMock()
    client.config = MagicMock(irrigation_channel_id="", irrigation_read_api_key=None)
    client.fetch_latest_environment_rows.return_value = [
        {"soil_moisture_percent": 30.0, "collected_at": "2025-06-01T12:00:00+00:00"},
    ]
    state = EnrichedFetchState()
    row = fetch_enriched_environment_row(client, state, env_history_rows=1)
    client.fetch_latest_irrigation_event_row.assert_not_called()
    assert row["_runtime_monitoring"].get("irrigation_context_skipped") == "channel_or_read_key_not_configured"
