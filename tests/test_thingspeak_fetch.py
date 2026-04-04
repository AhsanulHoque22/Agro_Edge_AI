"""Unit tests for ThingSpeak HTTP helpers (mocked)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from data_pipeline.ingestion.thingspeak_fetch import (
    fetch_feeds_chunk,
    fetch_feeds_timerange,
)


def test_fetch_feeds_chunk_builds_url_and_returns_feeds() -> None:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "feeds": [
            {
                "created_at": "2024-06-01T08:00:00Z",
                "entry_id": 42,
                "field1": "55.2",
            }
        ]
    }

    with patch("data_pipeline.ingestion.thingspeak_fetch.requests.get", return_value=resp) as get:
        feeds = fetch_feeds_chunk(
            base_url="https://api.thingspeak.com",
            channel_id="9",
            api_key="READKEY",
            start=datetime(2024, 6, 1, tzinfo=UTC),
            end=datetime(2024, 6, 2, tzinfo=UTC),
        )

    assert len(feeds) == 1
    assert feeds[0]["entry_id"] == 42
    get.assert_called_once()
    call_kw = get.call_args
    assert "channels/9/feeds.json" in call_kw[0][0]
    params = call_kw[1]["params"]
    assert params["api_key"] == "READKEY"
    assert "start" in params and "end" in params


def test_fetch_feeds_timerange_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="range_end"):
        fetch_feeds_timerange(
            base_url="https://api.thingspeak.com",
            channel_id="1",
            api_key="KEY",
            range_start=datetime(2024, 2, 1, tzinfo=UTC),
            range_end=datetime(2024, 1, 1, tzinfo=UTC),
            pause_seconds=0.0,
        )
