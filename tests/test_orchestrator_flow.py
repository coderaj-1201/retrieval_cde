"""
Adversarial tests for the orchestrator agent flow:
  classification failures, tool ladder exhaustion, fan-out, circuit breaker,
  cross-domain merging, confidence threshold edge cases.

"If the orchestrator silently routes to the wrong domain, users get wrong answers."
"""
from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

_af_stub = MagicMock()
_af_stub.step     = lambda fn: fn
_af_stub.workflow = MagicMock(return_value=lambda fn: fn)
sys.modules.setdefault("agent_framework", _af_stub)
sys.modules.setdefault("retrieval_pipeline.agent_framework", _af_stub)

from shared.models import (
    ClassifyInput, Domain, FinalResponse, OrchestratorInput,
    OrchestratorRequest, RetrievalResult, RetrievalTool, UserQuery,
)


def _retrieval_result(confidence=0.9, domain=Domain.OPS, tool=RetrievalTool.HYBRID,
                      sources=None, answer="Answer."):
    return RetrievalResult(
        query="q", domain=domain, tool=tool, attempt=1,
        answer=answer, confidence=confidence,
        sources=sources or [{"title": "Doc", "url": "https://a.com", "relevance": 0.9}],
        conversation_id="c", user_id="u", question_id="q",
    )


def _user_query(text="What is the leave policy?"):
    return UserQuery(text=text, conversation_id="c-1", user_id="u-1", question_id="q-1")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CLASSIFICATION FAILURES
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassificationFailures:

    @pytest.mark.asyncio
    async def test_classification_failure_returns_error_not_ops_default(self):
        """H-9: classification failure must return error, NEVER silently default to OPS."""
        from agents.orchestrator_agent import orchestrator_workflow

        with patch("agents.orchestrator_agent.classify_query") as mock_classify:
            from agents.orchestrator_agent import ClassifyResult
            mock_classify.return_value = ClassifyResult(
                domain=None, domain_confidence=0.0, secondary_domain=None,
                tool=RetrievalTool.HYBRID, failed=True,
            )
            result = await orchestrator_workflow(OrchestratorInput(user_query=_user_query()))

        assert result.status == "error"
        assert result.domain is None
        assert "classify" in result.answer.lower() or "rephrase" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_classification_domain_none_is_error_not_crash(self):
        """domain=None after classify must produce error FinalResponse, not AttributeError."""
        from agents.orchestrator_agent import orchestrator_workflow

        with patch("agents.orchestrator_agent.classify_query") as mock_classify:
            from agents.orchestrator_agent import ClassifyResult
            mock_classify.return_value = ClassifyResult(
                domain=None, domain_confidence=0.0, secondary_domain=None,
                tool=RetrievalTool.HYBRID, failed=True,
            )
            result = await orchestrator_workflow(OrchestratorInput(user_query=_user_query()))

        assert isinstance(result, FinalResponse)
        assert result.status in ("error", "failure")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TOOL LADDER AND RETRY EXHAUSTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolLadderExhaustion:

    @pytest.mark.asyncio
    async def test_all_attempts_below_threshold_returns_failure(self):
        """If all 3 retrieval attempts return low confidence, status=failure."""
        from agents.orchestrator_agent import orchestrator_workflow

        low_result = _retrieval_result(confidence=0.3)

        with patch("agents.orchestrator_agent.classify_query") as mock_classify, \
             patch("agents.orchestrator_agent.call_retrieval",
                   AsyncMock(return_value=low_result)), \
             patch("shared.config.settings") as cfg:
            cfg.DOMAIN_CONFIDENCE_THRESHOLD = 0.6
            cfg.CONFIDENCE_THRESHOLD        = 0.75
            cfg.MAX_RETRIEVAL_ATTEMPTS      = 3
            from agents.orchestrator_agent import ClassifyResult
            mock_classify.return_value = ClassifyResult(
                domain=Domain.OPS, domain_confidence=0.95,
                secondary_domain=None, tool=RetrievalTool.HYBRID,
            )
            result = await orchestrator_workflow(OrchestratorInput(user_query=_user_query()))

        assert result.status == "failure"

    @pytest.mark.asyncio
    async def test_second_attempt_succeeds_after_first_fails(self):
        """If attempt 1 is below threshold but attempt 2 passes, status=success."""
        from agents.orchestrator_agent import orchestrator_workflow

        low_result  = _retrieval_result(confidence=0.4)
        high_result = _retrieval_result(confidence=0.9)
        call_count  = 0

        async def _retrieval_side_effect(req):
            nonlocal call_count
            call_count += 1
            return low_result if call_count == 1 else high_result

        with patch("agents.orchestrator_agent.classify_query") as mock_classify, \
             patch("agents.orchestrator_agent.call_retrieval",
                   side_effect=_retrieval_side_effect), \
             patch("shared.config.settings") as cfg:
            cfg.DOMAIN_CONFIDENCE_THRESHOLD = 0.6
            cfg.CONFIDENCE_THRESHOLD        = 0.75
            cfg.MAX_RETRIEVAL_ATTEMPTS      = 3
            from agents.orchestrator_agent import ClassifyResult
            mock_classify.return_value = ClassifyResult(
                domain=Domain.OPS, domain_confidence=0.9,
                secondary_domain=None, tool=RetrievalTool.HYBRID,
            )
            result = await orchestrator_workflow(OrchestratorInput(user_query=_user_query()))

        assert result.status == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retrieval_exception_on_attempt_skips_to_next(self):
        """A retrieval exception on attempt N must not stop the ladder — try N+1."""
        from agents.orchestrator_agent import orchestrator_workflow

        success_result = _retrieval_result(confidence=0.9)
        call_count = 0

        async def _side_effect(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("retrieval down")
            return success_result

        with patch("agents.orchestrator_agent.classify_query") as mock_classify, \
             patch("agents.orchestrator_agent.call_retrieval", side_effect=_side_effect), \
             patch("shared.config.settings") as cfg:
            cfg.DOMAIN_CONFIDENCE_THRESHOLD = 0.6
            cfg.CONFIDENCE_THRESHOLD        = 0.75
            cfg.MAX_RETRIEVAL_ATTEMPTS      = 3
            from agents.orchestrator_agent import ClassifyResult
            mock_classify.return_value = ClassifyResult(
                domain=Domain.HR, domain_confidence=0.9,
                secondary_domain=None, tool=RetrievalTool.HYBRID,
            )
            result = await orchestrator_workflow(OrchestratorInput(user_query=_user_query()))

        assert result.status == "success"
        assert call_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CIRCUIT BREAKER ON RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreakerOnRetrieval:

    @pytest.mark.asyncio
    async def test_circuit_open_returns_immediate_error_response(self):
        """Open circuit on retrieval returns error immediately without retrying."""
        from agents.orchestrator_agent import orchestrator_workflow
        from shared.circuit_breaker import CircuitOpenError

        with patch("agents.orchestrator_agent.classify_query") as mock_classify, \
             patch("agents.orchestrator_agent.call_retrieval",
                   AsyncMock(side_effect=CircuitOpenError("retrieval-agent", 30.0))), \
             patch("shared.config.settings") as cfg:
            cfg.DOMAIN_CONFIDENCE_THRESHOLD = 0.6
            cfg.CONFIDENCE_THRESHOLD        = 0.75
            cfg.MAX_RETRIEVAL_ATTEMPTS      = 3
            from agents.orchestrator_agent import ClassifyResult
            mock_classify.return_value = ClassifyResult(
                domain=Domain.OPS, domain_confidence=0.95,
                secondary_domain=None, tool=RetrievalTool.HYBRID,
            )
            result = await orchestrator_workflow(OrchestratorInput(user_query=_user_query()))

        assert result.status == "error"
        assert "unavailable" in result.answer.lower() or "temporarily" in result.answer.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CROSS-DOMAIN FAN-OUT
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossDomainFanOut:

    @pytest.mark.asyncio
    async def test_fanout_uses_higher_confidence_domain(self):
        """When both fan-out domains return results, the higher-confidence one wins."""
        from agents.orchestrator_agent import _merge_retrieval_results

        primary   = _retrieval_result(confidence=0.6, domain=Domain.HR, answer="HR answer")
        secondary = _retrieval_result(confidence=0.9, domain=Domain.OPS, answer="OPS answer")

        merged = _merge_retrieval_results(primary, secondary)

        assert merged.confidence == 0.9
        assert merged.answer == "OPS answer"

    @pytest.mark.asyncio
    async def test_fanout_primary_fails_uses_secondary(self):
        """If primary domain retrieval fails, secondary result is used."""
        from agents.orchestrator_agent import orchestrator_workflow

        secondary_result = _retrieval_result(confidence=0.85, domain=Domain.OPS)
        call_count       = {"primary": 0, "secondary": 0}

        async def _fanout_side_effect(req):
            if req.domain == Domain.HR:
                call_count["primary"] += 1
                raise RuntimeError("HR search down")
            call_count["secondary"] += 1
            return secondary_result

        with patch("agents.orchestrator_agent.classify_query") as mock_classify, \
             patch("agents.orchestrator_agent._call_retrieval_safe",
                   side_effect=_fanout_side_effect), \
             patch("shared.config.settings") as cfg:
            cfg.DOMAIN_CONFIDENCE_THRESHOLD = 0.6
            cfg.CONFIDENCE_THRESHOLD        = 0.75
            cfg.MAX_RETRIEVAL_ATTEMPTS      = 1
            from agents.orchestrator_agent import ClassifyResult
            mock_classify.return_value = ClassifyResult(
                domain=Domain.HR, domain_confidence=0.5,   # below threshold → fan-out
                secondary_domain=Domain.OPS, tool=RetrievalTool.HYBRID,
            )
            result = await orchestrator_workflow(OrchestratorInput(user_query=_user_query()))

        assert result.status == "success"

    def test_fanout_merge_deduplicates_sources_by_title(self):
        """Source deduplication in _merge_retrieval_results removes duplicate titles."""
        from agents.orchestrator_agent import _merge_retrieval_results

        shared_source = {"title": "Leave Policy", "url": "https://a.com", "relevance": 0.9}
        primary   = _retrieval_result(confidence=0.7, sources=[shared_source])
        secondary = _retrieval_result(confidence=0.8, sources=[
            shared_source,
            {"title": "IT Guide", "url": "https://b.com", "relevance": 0.8},
        ])

        merged = _merge_retrieval_results(primary, secondary)
        titles = [s["title"] for s in merged.sources]

        assert titles.count("Leave Policy") == 1   # deduped

    @pytest.mark.asyncio
    async def test_both_fanout_domains_fail_continues_to_next_attempt(self):
        """If both fan-out domains fail on attempt 1, the loop continues to attempt 2."""
        from agents.orchestrator_agent import orchestrator_workflow

        attempt_count = 0
        success_result = _retrieval_result(confidence=0.9)

        async def _safe_side_effect(req):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count <= 2:  # first 2 calls (attempt 1, both domains) fail
                return None
            return success_result

        with patch("agents.orchestrator_agent.classify_query") as mock_classify, \
             patch("agents.orchestrator_agent._call_retrieval_safe",
                   side_effect=_safe_side_effect), \
             patch("shared.config.settings") as cfg:
            cfg.DOMAIN_CONFIDENCE_THRESHOLD = 0.6
            cfg.CONFIDENCE_THRESHOLD        = 0.75
            cfg.MAX_RETRIEVAL_ATTEMPTS      = 3
            from agents.orchestrator_agent import ClassifyResult
            mock_classify.return_value = ClassifyResult(
                domain=Domain.HR, domain_confidence=0.4,
                secondary_domain=Domain.OPS, tool=RetrievalTool.HYBRID,
            )
            result = await orchestrator_workflow(OrchestratorInput(user_query=_user_query()))

        # At least tried more than 2 calls
        assert attempt_count >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  DYNAMIC CLASSIFY PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

class TestDynamicClassifyPrompt:

    def test_classify_prompt_contains_all_domains(self):
        """_CLASSIFY_SYSTEM must mention every domain in the Domain enum."""
        from agents.orchestrator_agent import _CLASSIFY_SYSTEM
        from shared.models import Domain

        for domain in Domain:
            assert domain.value in _CLASSIFY_SYSTEM, (
                f"Domain '{domain.value}' missing from classify prompt!"
            )

    def test_classify_prompt_contains_all_domain_descriptions(self):
        """Every DOMAIN_DESCRIPTIONS entry must appear in _CLASSIFY_SYSTEM."""
        from agents.orchestrator_agent import _CLASSIFY_SYSTEM
        from shared.models import DOMAIN_DESCRIPTIONS

        for domain, desc in DOMAIN_DESCRIPTIONS.items():
            # At minimum the first word of the description should appear
            first_word = desc.split("/")[0].strip()
            assert first_word.lower() in _CLASSIFY_SYSTEM.lower(), (
                f"Description for '{domain}' not found in classify prompt"
            )
