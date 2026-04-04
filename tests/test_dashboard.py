"""Tests for minimal Flask dashboard."""

from __future__ import annotations

import json

import pytest

from web_dashboard.app import create_app


@pytest.fixture
def app_with_log(tmp_path, monkeypatch: pytest.MonkeyPatch):
    log = tmp_path / "cycles.jsonl"
    log.write_text(
        json.dumps(
            {
                "cycle_index": 1,
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "status": "ok",
                "decision": {"should_irrigate": False},
                "action_payload": {"node_id": "n1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGROEDGE_RUNTIME_LOG", str(log))
    monkeypatch.setenv("THINGSPEAK_ENV_CHANNEL_ID", "999")
    return create_app()


def test_health_ok():
    c = create_app().test_client()
    r = c.get("/health")
    assert r.status_code == 200
    assert r.data.strip() == b"ok"


def test_api_thingspeak_links(app_with_log):
    c = app_with_log.test_client()
    r = c.get("/api/thingspeak-links")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert any(link["channel_id"] == "999" for link in data["links"])


def test_api_last_cycle_ok(app_with_log):
    c = app_with_log.test_client()
    r = c.get("/api/last-cycle")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_index_contains_thingspeak_link(app_with_log):
    c = app_with_log.test_client()
    r = c.get("/")
    assert r.status_code == 200
    assert b"ThingSpeak" in r.data
    assert b"channels/999" in r.data
