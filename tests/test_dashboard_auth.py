"""Optional HTTP Basic auth on dashboard (excluding /health)."""

from __future__ import annotations

import json

import pytest

from web_dashboard.app import create_app


@pytest.fixture
def log_and_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
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


def test_health_exempt_when_basic_auth_enabled(log_and_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGROEDGE_DASHBOARD_BASIC_USER", "admin")
    monkeypatch.setenv("AGROEDGE_DASHBOARD_BASIC_PASSWORD", "secret")
    c = create_app().test_client()
    r = c.get("/health")
    assert r.status_code == 200


def test_api_requires_auth_when_configured(log_and_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGROEDGE_DASHBOARD_BASIC_USER", "admin")
    monkeypatch.setenv("AGROEDGE_DASHBOARD_BASIC_PASSWORD", "secret")
    c = create_app().test_client()
    r = c.get("/api/last-cycle")
    assert r.status_code == 401
    r2 = c.get("/api/last-cycle", auth=("admin", "secret"))
    assert r2.status_code == 200


def test_no_auth_when_env_unset(log_and_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AGROEDGE_DASHBOARD_BASIC_USER", raising=False)
    monkeypatch.delenv("AGROEDGE_DASHBOARD_BASIC_PASSWORD", raising=False)
    c = create_app().test_client()
    assert c.get("/api/last-cycle").status_code == 200
