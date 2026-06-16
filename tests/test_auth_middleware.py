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

def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(InternalAuthMiddleware)

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.post("/protected")
    async def protected():
        return {"data": "secret"}

    return app


def _mock_settings(secret: str | None) -> MagicMock:
    ms = MagicMock()
    if secret:
        ms.INTERNAL_API_SECRET = MagicMock()
        ms.INTERNAL_API_SECRET.get_secret_value.return_value = secret
    else:
        ms.INTERNAL_API_SECRET = None
    return ms


def test_health_exempt_no_secret_required():
    app = _build_app()
    with patch("shared.auth_middleware.settings", _mock_settings("supersecret")):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health/live")
    assert resp.status_code == 200


def test_protected_with_correct_secret():
    app = _build_app()
    with patch("shared.auth_middleware.settings", _mock_settings("supersecret")):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/protected", headers={"X-Internal-Secret": "supersecret"})
    assert resp.status_code == 200


def test_protected_with_wrong_secret():
    app = _build_app()
    with patch("shared.auth_middleware.settings", _mock_settings("supersecret")):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/protected", headers={"X-Internal-Secret": "wrong"})
    assert resp.status_code == 401


def test_protected_with_no_secret_header():
    app = _build_app()
    with patch("shared.auth_middleware.settings", _mock_settings("supersecret")):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/protected")
    assert resp.status_code == 401


def test_secret_not_configured_allows_all():
    """When INTERNAL_API_SECRET is not set, middleware logs a warning and passes traffic."""
    app = _build_app()
    with patch("shared.auth_middleware.settings", _mock_settings(None)):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/protected")
    assert resp.status_code == 200
