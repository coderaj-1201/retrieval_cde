"""Unit tests for card_mapper.py — no Azure dependencies."""
from __future__ import annotations

import pytest
from card_mapper import build_answer_card, build_feedback_card, normalize_sources, _safe_url


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
