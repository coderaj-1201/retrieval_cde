"""
Health probe tests — verifies liveness/readiness split is correctly implemented
across all four services, and that degraded dependencies produce 503.

"ACA removes unhealthy replicas from the load balancer based on these probes.
If they lie, traffic goes to broken pods."
"""
from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

_af_stub = MagicMock()
_af_stub.step     = lambda fn: fn
_af_stub.workflow = MagicMock(return_value=lambda fn: fn)
sys.modules.setdefault("agent_framework", _af_stub)
sys.modules.setdefault("retrieval_pipeline.agent_framework", _af_stub)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  TEAMS BOT HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

class TestTeamsBotHealth:

    def test_liveness_always_200(self):
        from fastapi.testclient import TestClient
        from teams_bot import app
        client = TestClient(app)
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_health_alias_returns_200(self):
        from fastapi.testclient import TestClient
        from teams_bot import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MAIN AGENT HEALTH PROBES
# ═══════════════════════════════════════════════════════════════════════════════

class TestMainAgentHealth:

    def test_liveness_always_200_even_when_cosmos_down(self):
        """Liveness must return 200 regardless of Cosmos state."""
        from fastapi.testclient import TestClient
        import agents.main_agent as ma
        client = TestClient(ma.app, raise_server_exceptions=False)
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_readiness_503_when_cosmos_unavailable(self):
        """If Cosmos is down, /health/ready must return 503."""
        from fastapi.testclient import TestClient
        import agents.main_agent as ma

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("agents.main_agent.get_chat_container") as mock_cc, \
             patch("shared.azure_clients.get_openai_client") as mock_oai, \
             patch("agents.main_agent._http") as mock_http:
            mock_cc.return_value.read.side_effect = Exception("Cosmos unreachable")
            mock_oai.return_value.models.list.return_value = MagicMock()
            mock_http.get = AsyncMock(return_value=mock_resp)

            client = TestClient(ma.app, raise_server_exceptions=False)
            resp = client.get("/health/ready")

        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] in ("degraded", "unhealthy")
        assert "cosmos" in data["checks"]
        assert data["checks"]["cosmos"].startswith("error")

    def test_readiness_200_when_all_healthy(self):
        """When all deps respond, /health/ready must return 200."""
        from fastapi.testclient import TestClient
        import agents.main_agent as ma

        mock_live_resp = MagicMock()
        mock_live_resp.status_code = 200

        with patch("agents.main_agent.get_chat_container") as mock_cc, \
             patch("shared.azure_clients.get_openai_client") as mock_oai, \
             patch("agents.main_agent._orchestrator_breaker") as mock_cb, \
             patch("agents.main_agent._http") as mock_http:
            mock_cc.return_value.read.return_value = None
            mock_oai.return_value.models.list.return_value = MagicMock()
            mock_cb.to_dict.return_value = {"state": "closed"}
            mock_http.get = AsyncMock(return_value=mock_live_resp)

            client = TestClient(ma.app, raise_server_exceptions=False)
            resp = client.get("/health/ready")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_readiness_503_when_orchestrator_circuit_open(self):
        """Open circuit to orchestrator must cause /health/ready to return 503."""
        from fastapi.testclient import TestClient
        import agents.main_agent as ma

        mock_live_resp = MagicMock()
        mock_live_resp.status_code = 200

        with patch("agents.main_agent.get_chat_container") as mock_cc, \
             patch("shared.azure_clients.get_openai_client") as mock_oai, \
             patch("agents.main_agent._orchestrator_breaker") as mock_cb, \
             patch("agents.main_agent._http") as mock_http:
            mock_cc.return_value.read.return_value = None
            mock_oai.return_value.models.list.return_value = MagicMock()
            mock_cb.to_dict.return_value = {"state": "open"}   # circuit open!
            mock_http.get = AsyncMock(return_value=mock_live_resp)

            client = TestClient(ma.app, raise_server_exceptions=False)
            resp = client.get("/health/ready")

        assert resp.status_code == 503
        assert resp.json()["checks"]["orchestrator_circuit"] == "open"


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ORCHESTRATOR HEALTH PROBES
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorHealth:

    def test_liveness_always_200(self):
        from fastapi.testclient import TestClient
        from agents.orchestrator_agent import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_readiness_503_when_retrieval_circuit_open(self):
        from fastapi.testclient import TestClient
        from agents.orchestrator_agent import app

        with patch("agents.orchestrator_agent._retrieval_breaker") as mock_cb, \
             patch("shared.cosmos_client.get_chat_container") as mock_cc:
            mock_cb.to_dict.return_value = {"state": "open"}
            mock_cc.return_value.read.return_value = None
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health/ready")

        assert resp.status_code == 503

    def test_readiness_200_when_circuit_closed_and_cosmos_ok(self):
        from fastapi.testclient import TestClient
        from agents.orchestrator_agent import app

        with patch("agents.orchestrator_agent._retrieval_breaker") as mock_cb, \
             patch("shared.cosmos_client.get_chat_container") as mock_cc:
            mock_cb.to_dict.return_value = {"state": "closed"}
            mock_cc.return_value.read.return_value = None
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health/ready")

        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  RETRIEVAL AGENT HEALTH PROBES
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalHealth:

    def test_liveness_always_200(self):
        from fastapi.testclient import TestClient
        from agents.retrieval_agent import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_readiness_503_when_cosmos_error(self):
        from fastapi.testclient import TestClient
        from agents.retrieval_agent import app

        with patch("shared.cosmos_client.get_chat_container") as mock_cc, \
             patch("agents.retrieval_agent.get_openai_client") as mock_oai:
            mock_cc.return_value.read.side_effect = Exception("Cosmos down")
            mock_oai.return_value.models.list.return_value = MagicMock()
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health/ready")

        assert resp.status_code == 503
        assert "cosmos" in resp.json()["checks"]

    def test_readiness_503_when_openai_error(self):
        from fastapi.testclient import TestClient
        from agents.retrieval_agent import app

        with patch("shared.cosmos_client.get_chat_container") as mock_cc, \
             patch("agents.retrieval_agent.get_openai_client") as mock_oai:
            mock_cc.return_value.read.return_value = None
            mock_oai.return_value.models.list.side_effect = Exception("OpenAI unreachable")
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health/ready")

        assert resp.status_code == 503
        assert resp.json()["checks"]["openai"].startswith("error")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  INTERNAL AUTH MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════

class TestInternalAuthOnAgents:

    def _agent_client(self, agent_module_path: str, secret: str | None = "s3cr3t"):
        """Create a test client for an agent with patched INTERNAL_API_SECRET."""
        import importlib
        mod = importlib.import_module(agent_module_path)

        from fastapi.testclient import TestClient

        mock_settings = MagicMock()
        if secret:
            mock_settings.INTERNAL_API_SECRET = MagicMock()
            mock_settings.INTERNAL_API_SECRET.get_secret_value.return_value = secret
        else:
            mock_settings.INTERNAL_API_SECRET = None

        with patch("shared.auth_middleware.settings", mock_settings):
            return TestClient(mod.app, raise_server_exceptions=False)

    def test_orchestrator_protected_endpoint_requires_secret(self):
        from fastapi.testclient import TestClient
        from agents.orchestrator_agent import app

        mock_settings = MagicMock()
        mock_settings.INTERNAL_API_SECRET = MagicMock()
        mock_settings.INTERNAL_API_SECRET.get_secret_value.return_value = "s3cr3t"

        with patch("shared.auth_middleware.settings", mock_settings):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/orchestrate",
                json={"text": "test", "conversation_id": "c", "user_id": "u", "question_id": "q"},
                headers={"X-Internal-Secret": "wrong-secret"},
            )
            assert resp.status_code == 401

    def test_orchestrator_health_exempt_from_auth(self):
        """Health endpoints must always be accessible without auth header."""
        from fastapi.testclient import TestClient
        from agents.orchestrator_agent import app

        mock_settings = MagicMock()
        mock_settings.INTERNAL_API_SECRET = MagicMock()
        mock_settings.INTERNAL_API_SECRET.get_secret_value.return_value = "s3cr3t"

        with patch("shared.auth_middleware.settings", mock_settings):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health/live")   # no auth header
            assert resp.status_code == 200

    def test_retrieval_protected_endpoint_requires_secret(self):
        from fastapi.testclient import TestClient
        from agents.retrieval_agent import app

        mock_settings = MagicMock()
        mock_settings.INTERNAL_API_SECRET = MagicMock()
        mock_settings.INTERNAL_API_SECRET.get_secret_value.return_value = "s3cr3t"

        with patch("shared.auth_middleware.settings", mock_settings):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/retrieve",
                json={"query": "test", "domain": "ops", "tool": "hybrid",
                      "attempt": 1, "conversation_id": "c", "user_id": "u"},
            )
            assert resp.status_code == 401
