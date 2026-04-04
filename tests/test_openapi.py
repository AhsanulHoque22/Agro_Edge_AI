"""OpenAPI routes for the dashboard."""

from __future__ import annotations

from web_dashboard.app import create_app


def test_openapi_json():
    c = create_app().test_client()
    r = c.get("/api/openapi.json")
    assert r.status_code == 200
    data = r.get_json()
    assert data.get("openapi") == "3.0.3"
    assert "paths" in data
    assert "/health" in data["paths"]
    assert "components" in data
    assert "dashboardBasic" in data["components"].get("securitySchemes", {})


def test_openapi_yaml():
    c = create_app().test_client()
    r = c.get("/api/openapi.yaml")
    assert r.status_code == 200
    assert b"openapi: 3.0.3" in r.data
    assert r.mimetype in ("application/yaml", "text/yaml") or "yaml" in (r.mimetype or "")
