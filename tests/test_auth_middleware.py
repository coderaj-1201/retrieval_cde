"""Unit tests for shared/auth_middleware.py — no Azure dependencies."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.auth_middleware import InternalAuthMiddleware, _EXEMPT_PATHS


# ── Exempt paths ───────────────────────────────────────────────────────────────

def test_exempt_paths_contains_health():
    assert "/health/live"  in _EXEMPT_PATHS
    assert "/health/ready" in _EXEMPT_PATHS
    assert "/health"       in _EXEMPT_PATHS


# ── Middleware behaviour ───────────────────────────────────────────────────────

def _make_app(secret: str | None) -> TestClient:
    app = FastAPI()
    app.add_middleware(InternalAuthMiddleware)

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.post("/protected")
    async def protected():
        return {"data": "secret"}

    mock_settings = MagicMock()
    if secret:
        mock_settings.INTERNAL_API_SECRET = MagicMock()
        mock_settings.INTERNAL_API_SECRET.get_secret_value.return_value = secret
    else:
        mock_settings.INTERNAL_API_SECRET = None

    with patch("shared.auth_middleware.settings", mock_settings):
        return TestClient(app, raise_server_exceptions=False)


def test_health_exempt_no_secret_required():
    client = _make_app(secret="supersecret")
    resp = client.get("/health/live")
    assert resp.status_code == 200


def test_protected_with_correct_secret():
    client = _make_app(secret="supersecret")
    resp = client.post("/protected", headers={"X-Internal-Secret": "supersecret"})
    assert resp.status_code == 200


def test_protected_with_wrong_secret():
    client = _make_app(secret="supersecret")
    resp = client.post("/protected", headers={"X-Internal-Secret": "wrong"})
    assert resp.status_code == 401


def test_protected_with_no_secret_header():
    client = _make_app(secret="supersecret")
    resp = client.post("/protected")
    assert resp.status_code == 401


def test_secret_not_configured_allows_all():
    """When INTERNAL_API_SECRET is not set, middleware logs a warning and passes traffic."""
    client = _make_app(secret=None)
    resp = client.post("/protected")
    assert resp.status_code == 200
