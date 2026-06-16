"""Unit tests for shared/models.py — no Azure dependencies."""
from __future__ import annotations

import pytest
from shared.models import (
    DOMAIN_DESCRIPTIONS, Domain, FinalResponse, QueryResponse,
    RetrievalResult, RetrievalTool,
)


# ── DOMAIN_DESCRIPTIONS ────────────────────────────────────────────────────────

def test_domain_descriptions_covers_all_domains():
    for domain in Domain:
        assert domain in DOMAIN_DESCRIPTIONS, f"DOMAIN_DESCRIPTIONS missing entry for {domain}"


def test_domain_descriptions_non_empty():
    for domain, desc in DOMAIN_DESCRIPTIONS.items():
        assert desc.strip(), f"Empty description for domain {domain}"


# ── RetrievalResult ────────────────────────────────────────────────────────────

def test_retrieval_result_passed_above_threshold():
    from unittest.mock import patch
    r = RetrievalResult(
        query="q", domain=Domain.HR, tool=RetrievalTool.HYBRID,
        attempt=1, answer="a", confidence=0.9, sources=[],
        conversation_id="c", user_id="u",
    )
    with patch("shared.config.settings") as m:
        m.CONFIDENCE_THRESHOLD = 0.75
        assert r.passed is True


def test_retrieval_result_failed_below_threshold():
    from unittest.mock import patch
    r = RetrievalResult(
        query="q", domain=Domain.IT, tool=RetrievalTool.HYDE,
        attempt=1, answer="a", confidence=0.4, sources=[],
        conversation_id="c", user_id="u",
    )
    with patch("shared.config.settings") as m:
        m.CONFIDENCE_THRESHOLD = 0.75
        assert r.passed is False


def test_retrieval_result_to_dict():
    r = RetrievalResult(
        query="test", domain=Domain.OPS, tool=RetrievalTool.DECOMPOSITION,
        attempt=2, answer="answer", confidence=0.85, sources=[{"title": "Doc"}],
        conversation_id="conv-1", user_id="user-1", question_id="q-1",
    )
    d = r.to_dict()
    assert d["domain"]  == "ops"
    assert d["tool"]    == "decomposition"
    assert d["attempt"] == 2


# ── FinalResponse ──────────────────────────────────────────────────────────────

def test_final_response_to_dict_with_none_domain():
    r = FinalResponse(status="error", answer="", domain=None)
    d = r.to_dict()
    assert d["domain"] == ""


def test_final_response_to_dict_with_domain():
    r = FinalResponse(status="success", answer="hello", domain=Domain.LEGAL)
    d = r.to_dict()
    assert d["domain"] == "legal"
    assert d["status"] == "success"


# ── QueryResponse ─────────────────────────────────────────────────────────────

def test_query_response_to_dict_round_trip():
    r = QueryResponse(
        question_id="q-1", answer_id="a-1", conversation_id="c-1",
        user_id="u-1", status="success", answer="The answer.",
        domain="HR", confidence=0.88, attempts_used=1,
        tools_used=["hybrid"], sources=[], escalation_options=None,
    )
    d = r.to_dict()
    assert d["question_id"] == "q-1"
    assert d["confidence"]  == 0.88
    assert d["timestamp"]   # auto-generated, must be non-empty
