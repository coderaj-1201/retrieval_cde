"""Unit tests for card_mapper.py — no Azure dependencies."""
from __future__ import annotations

import pytest
from card_mapper import (
    build_answer_card,
    build_escalation_card,
    build_escalation_confirmation_card,
    build_feedback_card,
    normalize_sources,
    _safe_url,
)


# ── _safe_url ──────────────────────────────────────────────────────────────────

def test_safe_url_none_returns_none():
    assert _safe_url(None) is None


def test_safe_url_empty_returns_none():
    assert _safe_url("") is None


def test_safe_url_clean_url_unchanged():
    url = "https://company.sharepoint.com/sites/ops/doc.pdf"
    assert _safe_url(url) == url


def test_safe_url_encodes_spaces():
    result = _safe_url("https://company.sharepoint.com/sites/my doc.pdf")
    assert " " not in result
    assert "%20" in result


def test_safe_url_preserves_query_string():
    url = "https://example.com/search?q=foo&page=1"
    assert _safe_url(url) == url


# ── normalize_sources ──────────────────────────────────────────────────────────

def test_normalize_empty():
    assert normalize_sources([]) == []
    assert normalize_sources(None) == []   # type: ignore[arg-type]


def test_normalize_deduplicates_by_title():
    sources = [
        {"title": "HR Policy", "url": "https://a.com/hr.pdf"},
        {"title": "HR Policy", "url": "https://b.com/hr2.pdf"},
    ]
    result = normalize_sources(sources)
    assert len(result) == 1
    assert result[0]["title"] == "HR Policy"


def test_normalize_deduplicates_by_url():
    sources = [
        {"title": "Doc A", "url": "https://same.com/doc.pdf"},
        {"title": "Doc B", "url": "https://same.com/doc.pdf"},
    ]
    result = normalize_sources(sources)
    assert len(result) == 1


def test_normalize_keeps_distinct_sources():
    sources = [
        {"title": "HR Policy",  "url": "https://a.com/hr.pdf"},
        {"title": "IT Runbook", "url": "https://b.com/it.pdf"},
    ]
    assert len(normalize_sources(sources)) == 2


def test_normalize_fallback_title():
    sources = [{"url": "https://x.com/doc.pdf"}]
    result = normalize_sources(sources)
    assert result[0]["title"] == "Source 1"


def test_normalize_url_encoded_in_output():
    sources = [{"title": "Doc", "url": "https://a.com/my file.pdf"}]
    result = normalize_sources(sources)
    assert " " not in result[0]["url"]


# ── build_answer_card ──────────────────────────────────────────────────────────

def test_build_answer_card_structure():
    data = {"answer": "Hello world", "sources": []}
    card = build_answer_card(data)
    assert card["contentType"] == "application/vnd.microsoft.card.adaptive"
    body = card["content"]["body"]
    assert body[0]["text"] == "Hello world"


def test_build_answer_card_with_sources():
    data = {
        "answer": "See policy below.",
        "sources": [{"title": "Leave Policy", "url": "https://a.com/leave.pdf"}],
    }
    card = build_answer_card(data)
    body_texts = [b.get("text", "") for b in card["content"]["body"]]
    assert any("Leave Policy" in t for t in body_texts)


def test_build_answer_card_url_in_link():
    data = {
        "answer": "Answer.",
        "sources": [{"title": "Doc", "url": "https://a.com/my doc.pdf"}],
    }
    card = build_answer_card(data)
    body_texts = " ".join(b.get("text", "") for b in card["content"]["body"])
    assert " " not in body_texts.split("(")[1].split(")")[0]   # URL part has no space


# ── build_feedback_card ────────────────────────────────────────────────────────

# ── build_escalation_card ────────────────────────────────────────────────────

def _escalation_data(**overrides):
    data = {
        "answer": "I wasn't able to find a confident answer.",
        "question_id": "q-1", "answer_id": "ans-1",
        "conversation_id": "c-1", "user_id": "u-1", "domain": "hr",
        "escalation_options": {
            "raise_ticket": {"sla": "4 business hours"},
            "connect_sme":  {"sla": "2 business hours"},
        },
    }
    data.update(overrides)
    return data


def test_build_escalation_card_has_both_buttons():
    card = build_escalation_card(_escalation_data(), question_text="What is my leave balance?")
    actions = card["content"]["actions"]
    types_data = [a["data"]["escalation_type"] for a in actions]
    assert "raise_ticket" in types_data
    assert "connect_sme" in types_data


def test_build_escalation_card_button_carries_context():
    card = build_escalation_card(_escalation_data(), question_text="What is my leave balance?")
    ticket_action = next(a for a in card["content"]["actions"] if a["data"]["escalation_type"] == "raise_ticket")
    assert ticket_action["data"]["action"] == "escalate"
    assert ticket_action["data"]["question_id"] == "q-1"
    assert ticket_action["data"]["domain"] == "hr"
    assert ticket_action["data"]["question_text"] == "What is my leave balance?"


def test_build_escalation_card_no_actions_without_options():
    card = build_escalation_card(_escalation_data(escalation_options={}))
    assert card["content"]["actions"] == []


# ── build_escalation_confirmation_card ──────────────────────────────────────

def test_confirmation_card_ticket_raised():
    data = {
        "status": "ticket_raised", "correlation_id": "REF-ABC123",
        "domain": "hr", "answer": "Ticket raised.",
    }
    card = build_escalation_confirmation_card(data)
    assert card["contentType"] == "application/vnd.microsoft.card.adaptive"
    # title and reference appear in body
    all_text = str(card["content"]["body"])
    assert "Support Ticket Raised" in all_text
    assert "REF-ABC123" in all_text
    assert "4 business hours" in all_text


def test_confirmation_card_sme_connecting():
    data = {
        "status": "sme_connecting", "correlation_id": "REF-DEF456",
        "domain": "it", "answer": "SME requested.",
    }
    card = build_escalation_confirmation_card(data)
    all_text = str(card["content"]["body"])
    assert "SME Connection Requested" in all_text
    assert "REF-DEF456" in all_text
    assert "2 business hours" in all_text


def test_confirmation_card_no_correlation_id_falls_back_to_answer():
    data = {"status": "ticket_raised", "correlation_id": "", "answer": "Contact support."}
    card = build_escalation_confirmation_card(data)
    all_text = str(card["content"]["body"])
    assert "Contact support." in all_text


def test_build_feedback_card_structure():
    data = {
        "question_id": "q-abc",
        "answer_id":   "ans-xyz",
        "conversation_id": "c-1",
        "user_id": "u-1",
    }
    card = build_feedback_card(data)
    assert card["contentType"] == "application/vnd.microsoft.card.adaptive"
    actions = card["content"]["actions"]
    titles = [a["title"] for a in actions]
    assert "👍" in titles
    assert "👎" in titles
