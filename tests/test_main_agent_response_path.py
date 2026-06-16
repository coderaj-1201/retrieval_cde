"""
Regression test for the call_orchestrator bug where response parsing was placed
after `raise` and was unreachable, causing every query to return None.

Tests verify that _do_orchestrate + response parsing produce a correct FinalResponse
on the success path without calling any Azure services.
"""
from __future__ import annotations

import asyncio
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.models import Domain, FinalResponse, OrchestratorInput, UserQuery


def _make_orchestrator_payload(status="success", confidence=0.9, domain="ops"):
    return {
        "status":          status,
        "answer":          "Test answer.",
        "domain":          domain,
        "sources":         [{"title": "Doc A", "url": "https://a.com/doc.pdf"}],
        "confidence":      confidence,
        "attempts_used":   1,
        "conversation_id": "conv-1",
        "user_id":         "user-1",
        "question_id":     "q-1",
        "answer_id":       "ans-abc",
        "tools_used":      ["hybrid"],
    }


@pytest.mark.asyncio
async def test_call_orchestrator_success_returns_final_response():
    """Success path must return a FinalResponse, not None."""
    from agents.main_agent import call_orchestrator, _orchestrator_breaker

    user_query = UserQuery(
        text="What is the leave policy?",
        conversation_id="conv-1",
        user_id="user-1",
        question_id="q-1",
    )
    inp = OrchestratorInput(user_query=user_query)
    mock_data = _make_orchestrator_payload()

    with patch("agents.main_agent._do_orchestrate", new=AsyncMock(return_value=mock_data)), \
         patch("agents.main_agent._orchestrator_breaker") as mock_breaker:
        mock_breaker.call = AsyncMock(return_value=mock_data)
        result = await call_orchestrator(inp)

    assert result is not None, "call_orchestrator returned None on success path — regression!"
    assert isinstance(result, FinalResponse)
    assert result.status    == "success"
    assert result.answer    == "Test answer."
    assert result.domain    == Domain.OPS
    assert result.confidence == 0.9
    assert result.answer_id  == "ans-abc"


@pytest.mark.asyncio
async def test_call_orchestrator_unknown_domain_is_none():
    from agents.main_agent import call_orchestrator
    from shared.circuit_breaker import CircuitBreaker

    user_query = UserQuery(
        text="Some query", conversation_id="c", user_id="u", question_id="q"
    )
    inp = OrchestratorInput(user_query=user_query)
    mock_data = _make_orchestrator_payload(domain="unknown_xyz")

    with patch("agents.main_agent._orchestrator_breaker") as mock_breaker:
        mock_breaker.call = AsyncMock(return_value=mock_data)
        result = await call_orchestrator(inp)

    assert result is not None
    assert result.domain is None   # unrecognised domain must be None, not raise


@pytest.mark.asyncio
async def test_call_orchestrator_reraises_circuit_open():
    from agents.main_agent import call_orchestrator
    from shared.circuit_breaker import CircuitOpenError

    user_query = UserQuery(
        text="Query", conversation_id="c", user_id="u", question_id="q"
    )
    inp = OrchestratorInput(user_query=user_query)

    with patch("agents.main_agent._orchestrator_breaker") as mock_breaker:
        mock_breaker.call = AsyncMock(side_effect=CircuitOpenError("orchestrator", 30.0))
        with pytest.raises(CircuitOpenError):
            await call_orchestrator(inp)
