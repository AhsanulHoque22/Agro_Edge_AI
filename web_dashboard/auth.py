"""
Optional HTTP Basic authentication for the monitoring dashboard.

Enable with ``AGROEDGE_DASHBOARD_BASIC_USER`` and
``AGROEDGE_DASHBOARD_BASIC_PASSWORD`` in the environment (e.g. systemd
``EnvironmentFile``). Use only behind TLS (reverse proxy or local LAN with care).

``/health`` stays unauthenticated so load balancers and systemd probes still work.
"""

from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING

from flask import Response

if TYPE_CHECKING:
    from flask import Flask


def basic_auth_config_from_env() -> tuple[str, str] | None:
    user = os.getenv("AGROEDGE_DASHBOARD_BASIC_USER", "").strip()
    password = os.getenv("AGROEDGE_DASHBOARD_BASIC_PASSWORD", "").strip()
    if not user or not password:
        return None
    return user, password


def apply_optional_basic_auth(app: Flask) -> None:
    """Register ``before_request`` when credentials are configured."""
    cfg = basic_auth_config_from_env()
    if cfg is None:
        return

    expected_user, expected_password = cfg

    @app.before_request
    def _require_basic_auth() -> Response | None:
        from flask import request

        if request.path.rstrip("/") == "/health":
            return None

        auth = request.authorization
        if auth is None:
            return _unauthorized()
        u_ok = secrets.compare_digest(auth.username or "", expected_user)
        p_ok = secrets.compare_digest(auth.password or "", expected_password)
        if not (u_ok and p_ok):
            return _unauthorized()
        return None


def _unauthorized() -> Response:
    return Response(
        "Authentication required\n",
        status=401,
        mimetype="text/plain",
        headers={"WWW-Authenticate": 'Basic realm="AgroEdge dashboard"'},
    )


def is_basic_auth_enabled() -> bool:
    return basic_auth_config_from_env() is not None
