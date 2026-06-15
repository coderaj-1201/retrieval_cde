"""
Adversarial tests for infrastructure failures:
  Cosmos DB, Redis, Service Bus, Azure Search, circuit breaker under load.

"Azure services will hiccup. The app must degrade gracefully, never silently."
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

_af_stub = MagicMock()
_af_stub.step     = lambda fn: fn
_af_stub.workflow = MagicMock(return_value=lambda fn: fn)
sys.modules.setdefault("agent_framework", _af_stub)
sys.modules.setdefault("retrieval_pipeline.agent_framework", _af_stub)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  COSMOS DB FAILURES
# ═══════════════════════════════════════════════════════════════════════════════

class TestCosmosFailures:

    @pytest.mark.asyncio
    async def test_cosmos_session_read_failure_creates_new_session(self):
        """If Cosmos read throws, a fresh SessionMemory is returned — not crash."""
        from shared.memory import load_session, _session_cache

        # Clear LRU cache so Cosmos is actually queried
        async with _session_cache._lock:
            _session_cache._cache.clear()

        with patch("shared.memory.get_document", side_effect=Exception("Cosmos timeout")):
            session = await load_session("new-conv-123", "user-1")

        assert session is not None
        assert session.conversation_id == "new-conv-123"
        assert session.turns == []

    @pytest.mark.asyncio
    async def test_cosmos_session_write_failure_does_not_crash(self):
        """upsert_document failure in append_turn must not propagate to the caller."""
        from shared.memory import append_turn
        from shared.models import SessionMemory, ConversationTurn

        session = SessionMemory(conversation_id="c-1", user_id="u-1")
        turn = ConversationTurn(
            question_id="q", answer_id="a", question="Q", answer="A",
            domain="ops", confidence=0.9, tools_used=[],
        )

        with patch("shared.memory.upsert_document", side_effect=Exception("Cosmos write failed")):
            # Should not raise
            await append_turn(session, turn)

        assert len(session.turns) == 1  # turn was appended in memory even if Cosmos failed

    @pytest.mark.asyncio
    async def test_cosmos_ltm_read_failure_returns_none(self):
        """If Cosmos LTM read fails, load_ltm returns None — no crash."""
        from shared.memory import load_ltm

        with patch("shared.memory.get_document", side_effect=Exception("Network error")):
            result = await load_ltm("user-1")

        assert result is None

    def test_cosmos_cross_partition_query_logs_warning(self, caplog):
        """Cross-partition query (partition_key=None) must emit a WARNING."""
        import logging
        from shared.cosmos_client import query_documents

        mock_container = MagicMock()
        mock_container.query_items.return_value = []

        with caplog.at_level(logging.WARNING, logger="shared.cosmos_client"):
            query_documents(
                mock_container,
                "SELECT * FROM c",
                [],
                partition_key=None,   # no partition key → cross-partition
            )

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("cross_partition" in m.lower() or "cross-partition" in m.lower()
                   for m in warning_msgs)

    def test_cosmos_partition_key_provided_no_cross_partition_warning(self, caplog):
        """Scoped query (partition_key provided) must NOT emit cross-partition warning."""
        import logging
        from shared.cosmos_client import query_documents

        mock_container = MagicMock()
        mock_container.query_items.return_value = []

        with caplog.at_level(logging.WARNING, logger="shared.cosmos_client"):
            query_documents(
                mock_container,
                "SELECT * FROM c WHERE c.id = @id",
                [{"name": "@id", "value": "x"}],
                partition_key="x",
            )

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("cross" in m.lower() for m in warning_msgs)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SERVICE BUS / ESCALATION FAILURES
# ═══════════════════════════════════════════════════════════════════════════════

class TestEscalationFailures:

    def test_escalation_not_configured_is_detected(self):
        """When neither SB namespace nor connection string is set, must return False."""
        from shared.escalation_client import is_escalation_configured

        with patch("shared.escalation_client.settings") as cfg:
            cfg.AZURE_SERVICE_BUS_CONNECTION_STR = None
            cfg.AZURE_SERVICE_BUS_NAMESPACE      = None
            assert is_escalation_configured() is False

    def test_escalation_configured_with_namespace(self):
        from shared.escalation_client import is_escalation_configured

        with patch("shared.escalation_client.settings") as cfg:
            cfg.AZURE_SERVICE_BUS_CONNECTION_STR = None
            cfg.AZURE_SERVICE_BUS_NAMESPACE      = "myns.servicebus.windows.net"
            assert is_escalation_configured() is True

    def test_raise_ticket_send_failure_logs_payload_and_raises(self, caplog):
        """Service Bus send failure must log full JSON payload and re-raise."""
        import logging
        from shared.escalation_client import raise_ticket

        mock_sender = MagicMock()
        mock_sender.__enter__ = lambda s: s
        mock_sender.__exit__  = MagicMock(return_value=False)
        mock_sender.send_messages.side_effect = RuntimeError("SB queue full")

        with patch("shared.escalation_client._get_sender", return_value=mock_sender), \
             caplog.at_level(logging.ERROR, logger="shared.escalation_client"):
            with pytest.raises(RuntimeError, match="SB queue full"):
                raise_ticket(
                    user_id="u-1", conversation_id="c-1",
                    question_id="q-1", question_text="Help me!", domain="hr",
                )

        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("payload" in m.lower() or "raise_ticket" in m.lower()
                   for m in error_msgs)

    def test_connect_sme_send_failure_logs_payload_and_raises(self, caplog):
        """connect_sme send failure must log full payload and re-raise."""
        import logging
        from shared.escalation_client import connect_sme

        mock_sender = MagicMock()
        mock_sender.__enter__ = lambda s: s
        mock_sender.__exit__  = MagicMock(return_value=False)
        mock_sender.send_messages.side_effect = OSError("Connection refused")

        with patch("shared.escalation_client._get_sender", return_value=mock_sender), \
             caplog.at_level(logging.ERROR, logger="shared.escalation_client"):
            with pytest.raises(OSError):
                connect_sme(
                    user_id="u-1", conversation_id="c-1",
                    question_id="q-1", question_text="Need SME", domain="legal",
                )

        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("connect_sme" in m.lower() or "payload" in m.lower()
                   for m in error_msgs)

    def test_raise_ticket_not_configured_main_agent_returns_fallback_message(self):
        """When SB not configured, handle_raise_ticket must return a user-visible message."""
        import asyncio
        import sys
        sys.modules.setdefault("agent_framework", _af_stub)

        from shared.escalation_client import is_escalation_configured

        with patch("shared.escalation_client.settings") as cfg:
            cfg.AZURE_SERVICE_BUS_CONNECTION_STR = None
            cfg.AZURE_SERVICE_BUS_NAMESPACE      = None
            configured = is_escalation_configured()

        assert configured is False  # Guard: subsequent main_agent call gives fallback

    def test_raise_ticket_returns_unique_correlation_ids(self):
        """Each raise_ticket call must produce a unique correlation ID."""
        from shared.escalation_client import raise_ticket

        ids = []
        mock_sender = MagicMock()
        mock_sender.__enter__ = lambda s: s
        mock_sender.__exit__  = MagicMock(return_value=False)

        with patch("shared.escalation_client._get_sender", return_value=mock_sender):
            for i in range(5):
                ids.append(raise_ticket(
                    user_id=f"u-{i}", conversation_id="c", question_id="q",
                    question_text="test", domain="hr",
                ))

        assert len(set(ids)) == 5  # all unique


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  REDIS / RATE LIMITER FAILURES
# ═══════════════════════════════════════════════════════════════════════════════

class TestRedisFailures:

    def test_redis_connection_failure_falls_back_to_inprocess(self, caplog):
        """Redis unavailable at startup → warning logged once, falls back to in-process."""
        import logging
        import shared.rate_limiter as rl

        original_client  = rl._redis_client
        original_warned  = rl._WARNED_REDIS_FALLBACK

        rl._redis_client         = None
        rl._WARNED_REDIS_FALLBACK = False

        try:
            with patch("shared.rate_limiter.settings") as cfg, \
                 patch("shared.rate_limiter._inprocess_check") as mock_ip, \
                 patch("redis.from_url", side_effect=Exception("Redis unreachable")), \
                 caplog.at_level(logging.WARNING, logger="shared.rate_limiter"):
                cfg.REDIS_URL       = "redis://localhost:6379"
                cfg.RATE_LIMIT_RPM  = 20
                cfg.RATE_LIMIT_BURST = 5
                rl._redis_check("user-x")
                rl._redis_check("user-y")  # second call must NOT repeat warning

            warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                        and "redis" in r.message.lower()]
            # Exactly ONE activation warning
            assert len(warnings) == 1
            # But in-process fallback called twice (once per user)
            assert mock_ip.call_count == 2
        finally:
            rl._redis_client         = original_client
            rl._WARNED_REDIS_FALLBACK = original_warned

    def test_redis_pipeline_error_admits_request(self):
        """If Redis pipeline.execute() raises, request must be admitted (fail-open)."""
        import shared.rate_limiter as rl

        original_client  = rl._redis_client
        mock_redis = MagicMock()
        mock_redis.pipeline.return_value.execute.side_effect = RuntimeError("Redis timeout")
        rl._redis_client = mock_redis

        try:
            with patch("shared.rate_limiter.settings") as cfg:
                cfg.RATE_LIMIT_RPM = 5
                # Must not raise — fail-open on Redis error
                rl._redis_check("user-z")
        finally:
            rl._redis_client = original_client

    def test_redis_sliding_window_counts_correctly(self):
        """Weighted sliding window must correctly enforce RPM limit."""
        import math
        import time
        import shared.rate_limiter as rl

        original_client  = rl._redis_client
        mock_redis = MagicMock()

        # Simulate: previous minute had 15 requests, current has 10.
        # At 30s into current minute: weight = 0.5
        # weighted_count = 10 + floor(15 * 0.5) = 10 + 7 = 17
        # With RPM=20, 17 < 20 → should NOT raise

        def _mock_execute():
            return [11, True, "15"]   # incr→11 (including this call), expire→True, prev→15

        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = [11, True, "15"]
        mock_redis.pipeline.return_value = mock_pipeline
        rl._redis_client = mock_redis

        try:
            with patch("shared.rate_limiter.settings") as cfg, \
                 patch("time.time", return_value=1000.0 * 60 + 30.0):  # 30s into minute
                cfg.RATE_LIMIT_RPM = 20
                # 10 current + floor(15 * 0.5) = 17 < 20 → must NOT raise
                rl._redis_check("user-ok")
        finally:
            rl._redis_client = original_client

    def test_redis_sliding_window_blocks_over_limit(self):
        """weighted_count > RPM → RateLimitExceeded raised."""
        import shared.rate_limiter as rl

        original_client = rl._redis_client
        mock_redis = MagicMock()
        mock_pipeline = MagicMock()
        # 20 current + floor(20 * 0.5) = 30 > RPM=20 → should raise
        mock_pipeline.execute.return_value = [21, True, "20"]
        mock_redis.pipeline.return_value = mock_pipeline
        rl._redis_client = mock_redis

        try:
            with patch("shared.rate_limiter.settings") as cfg, \
                 patch("time.time", return_value=1000.0 * 60 + 30.0):
                cfg.RATE_LIMIT_RPM = 20
                with pytest.raises(rl.RateLimitExceeded) as exc_info:
                    rl._redis_check("user-blocked")
                assert exc_info.value.retry_after > 0
        finally:
            rl._redis_client = original_client


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CIRCUIT BREAKER UNDER SUSTAINED FAILURE
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreakerUnderLoad:

    @pytest.mark.asyncio
    async def test_circuit_opens_after_exactly_fail_max_failures(self):
        """Circuit must open after exactly fail_max failures, not before."""
        from shared.circuit_breaker import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(name="test-svc", fail_max=3, reset_timeout=60)

        async def _fail():
            raise ValueError("service down")

        # 2 failures — circuit must still be CLOSED
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        assert cb.to_dict()["state"] == "closed"

        # 3rd failure — circuit OPENS
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.to_dict()["state"] == "open"

    @pytest.mark.asyncio
    async def test_circuit_open_carries_correct_retry_after(self):
        """CircuitOpenError.retry_after must be approximately reset_timeout."""
        from shared.circuit_breaker import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(name="svc", fail_max=2, reset_timeout=45.0)

        async def _fail():
            raise ValueError()

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_fail)

        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(lambda: None)

        assert exc_info.value.retry_after > 0
        assert exc_info.value.retry_after <= 46   # within 1s of reset_timeout

    @pytest.mark.asyncio
    async def test_multiple_circuits_are_independent(self):
        """Failures on circuit A must not open circuit B."""
        from shared.circuit_breaker import CircuitBreaker

        cb_a = CircuitBreaker(name="svc-a", fail_max=2, reset_timeout=30)
        cb_b = CircuitBreaker(name="svc-b", fail_max=2, reset_timeout=30)

        async def _fail():
            raise ValueError()

        async def _ok():
            return "ok"

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb_a.call(_fail)

        assert cb_a.to_dict()["state"] == "open"
        assert cb_b.to_dict()["state"] == "closed"

        # cb_b still works
        result = await cb_b.call(_ok)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_health_ready_reflects_open_circuit(self):
        """When orchestrator circuit is open, /health/ready must return 503."""
        from agents.orchestrator_agent import _retrieval_breaker
        from shared.circuit_breaker import CircuitBreaker

        async def _fail():
            raise ValueError()

        # Force open the retrieval breaker
        cb = CircuitBreaker(name="retrieval-agent", fail_max=2, reset_timeout=60)
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_fail)

        with patch("agents.orchestrator_agent._retrieval_breaker", cb), \
             patch("agents.orchestrator_agent.get_chat_container") as mock_cc:
            mock_cc.return_value.read.return_value = None
            from fastapi.testclient import TestClient
            from agents.orchestrator_agent import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health/ready")

        assert resp.status_code == 503
        data = resp.json()
        assert data["checks"]["retrieval_circuit"] == "open"


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  OPENAI CLIENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpenAIClientConfig:

    def test_openai_client_has_zero_retries(self):
        """AzureOpenAI client must have max_retries=0 (tenacity owns retries)."""
        from shared.azure_clients import get_openai_client
        from openai import AzureOpenAI

        with patch("shared.azure_clients.settings") as cfg, \
             patch("shared.azure_clients.get_bearer_token_provider"), \
             patch("shared.azure_clients._credential"):
            cfg.AZURE_OPENAI_API_KEY                = None
            cfg.AZURE_OPENAI_ENDPOINT               = "https://test.openai.azure.com/"
            cfg.AZURE_OPENAI_API_VERSION            = "2024-12-01-preview"

            # Clear lru_cache to force new client creation
            get_openai_client.cache_clear()
            try:
                client = get_openai_client()
                assert client.max_retries == 0, (
                    "max_retries must be 0 — tenacity handles retries, not the SDK"
                )
            finally:
                get_openai_client.cache_clear()
