"""
Inter-agent authentication middleware.

Validates the X-Internal-Secret header on every inbound request to the
Orchestrator and Retrieval agents. Requests without a valid secret are
rejected with 401 before any processing occurs.

Health probe endpoints are always exempt so ACA probes keep working even
when INTERNAL_API_SECRET is not yet set in an environment.

Usage (in each internal agent's FastAPI app):
    from shared.auth_middleware import InternalAuthMiddleware
    app.add_middleware(InternalAuthMiddleware)

Callers (e.g. Main Agent → Orchestrator) must add the header:
    headers={"X-Internal-Secret": settings.INTERNAL_API_SECRET.get_secret_value()}
"""
from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from shared.config import settings

logger = logging.getLogger(__name__)

# Paths that are always accessible without auth — health probes and root.
_EXEMPT_PATHS = frozenset({"/health", "/health/live", "/health/ready", "/"})


class InternalAuthMiddleware(BaseHTTPMiddleware):
    """
    Validates X-Internal-Secret on all non-exempt paths.

    When INTERNAL_API_SECRET is not configured (local dev), auth is skipped
    with a one-time WARNING at startup rather than blocking all traffic.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        expected_secret = (
            settings.INTERNAL_API_SECRET.get_secret_value()
            if settings.INTERNAL_API_SECRET is not None
            else None
        )

        if not expected_secret:
            # Not configured — allow through but warn so it's visible in logs.
            logger.warning(
                "internal_auth_not_configured path=%s — "
                "INTERNAL_API_SECRET is not set. All requests are admitted.",
                request.url.path,
            )
            return await call_next(request)

        incoming = request.headers.get("X-Internal-Secret", "")
        if incoming != expected_secret:
            logger.warning(
                "internal_auth_rejected path=%s client=%s",
                request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "detail": "Invalid or missing X-Internal-Secret header."},
            )

        return await call_next(request)
