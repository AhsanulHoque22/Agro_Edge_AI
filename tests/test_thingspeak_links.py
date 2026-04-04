"""Tests for ThingSpeak web link helpers."""

from __future__ import annotations

import pytest

from web_dashboard.thingspeak_links import _channel_url, collect_configured_links


def test_collect_configured_links_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("THINGSPEAK_ENV_CHANNEL_ID", raising=False)
    monkeypatch.delenv("THINGSPEAK_IRRIGATION_CHANNEL_ID", raising=False)
    assert collect_configured_links() == []


def test_collect_configured_links_both(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("THINGSPEAK_ENV_CHANNEL_ID", "111")
    monkeypatch.setenv("THINGSPEAK_IRRIGATION_CHANNEL_ID", "222")
    links = collect_configured_links()
    assert len(links) == 2
    assert links[0].channel_id == "111"
    assert links[0].channel_url.endswith("/channels/111")
    assert links[1].channel_id == "222"


def test_channel_url_respects_web_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGROEDGE_THINGSPEAK_WEB_BASE", "https://example.com")
    assert _channel_url("55") == "https://example.com/channels/55"
