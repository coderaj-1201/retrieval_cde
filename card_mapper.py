"""
card_mapper.py
Two separate cards:
1. build_answer_card  — answer + sources only, no buttons
2. build_feedback_card — just 👍 👎 with collapsible comment
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote as _url_quote


def _safe_url(url: str | None) -> str | None:
    """Percent-encode a URL to make it safe for Adaptive Card markdown links.

    Preserves all URL structural characters so the link still resolves.
    Only encodes characters that would break the Markdown link syntax or
    cause Adaptive Card rendering issues (principally unencoded spaces and
    control characters).
    """
    if not url:
        return None
    # safe= keeps all valid URL chars intact; encodes spaces and control chars.
    return _url_quote(url, safe=":/?=&%#+@!$,;")


def normalize_sources(sources: Any) -> list[dict]:
    if not isinstance(sources, list):
        return []
    seen_titles: set[str] = set()
    seen_urls:   set[str] = set()
    result = []
    for i, s in enumerate(sources, start=1):
        if not isinstance(s, dict):
            continue
        title = (
            s.get("title") or s.get("name") or s.get("document_name")
            or s.get("documentName") or f"Source {i}"
        )
        raw_url = s.get("url") or s.get("source_url") or s.get("sourceUrl")
        # Deduplicate by title AND by URL — same document may appear under
        # different chunk titles when parent + child chunks are merged.
        if title in seen_titles:
            continue
        if raw_url and raw_url in seen_urls:
            continue
        seen_titles.add(title)
        if raw_url:
            seen_urls.add(raw_url)
        result.append({
            "title": title,
            "url":   _safe_url(raw_url),
            "page":  s.get("page") or s.get("page_number"),
        })
    return result


def _confidence_badge(score: float) -> str:
    """Return a short human-readable confidence label for a citation."""
    pct = int(round(score * 100))
    if pct >= 80:
        return f"🟢 {pct}%"
    if pct >= 60:
        return f"🟡 {pct}%"
    return f"🔴 {pct}%"


def build_answer_card(agent_response: dict) -> dict:
    """Answer card with conditional citation block.

    Citation logic (mirrors the LLM prompt contract):
      - show_citations = True  → render citations with per-doc confidence badge
      - show_citations = False → render answer only (greetings, low-confidence, errors)
    """
    answer        = (agent_response.get("answer") or "").strip()
    show_citations = bool(agent_response.get("show_citations", False))
    # LLM-generated citations (title + confidence + excerpt) take priority.
    # Fall back to search-result sources if citations list is absent.
    llm_citations: list[dict] = agent_response.get("citations") or []

    body: list[dict] = [
        {"type": "TextBlock", "text": answer, "wrap": True, "spacing": "None"},
    ]

    if show_citations and llm_citations:
        body.append({
            "type": "TextBlock", "text": "**Sources**",
            "wrap": True, "size": "Small", "weight": "Bolder",
            "spacing": "Medium", "separator": True,
        })
        for i, cite in enumerate(llm_citations[:5], start=1):
            title   = cite.get("title") or f"Source {i}"
            score   = float(cite.get("confidence", 0.0))
            excerpt = (cite.get("excerpt") or "").strip()
            badge   = _confidence_badge(score)

            # Title line with confidence badge
            body.append({
                "type": "TextBlock",
                "text": f"{i}. **{title}** — {badge}",
                "wrap": True, "size": "Small", "spacing": "Small",
            })
            # Excerpt line (subtle)
            if excerpt:
                body.append({
                    "type": "TextBlock",
                    "text": excerpt,
                    "wrap": True, "size": "Small", "isSubtle": True, "spacing": "None",
                })

    elif show_citations and not llm_citations:
        # LLM said show citations but returned none — fall back to search sources
        sources = normalize_sources(agent_response.get("sources", []))
        if sources:
            body.append({
                "type": "TextBlock", "text": "**Sources**",
                "wrap": True, "size": "Small", "weight": "Bolder",
                "spacing": "Medium", "separator": True,
            })
            for i, src in enumerate(sources[:5], start=1):
                page  = f" · Page {src['page']}" if src.get("page") else ""
                title = src["title"]
                url   = src.get("url")
                line  = f"[{i}. {title}]({url}){page}" if url else f"{i}. {title}{page}"
                body.append({
                    "type": "TextBlock", "text": line,
                    "wrap": True, "size": "Small", "isSubtle": True, "spacing": "Small",
                })

    # show_citations = False → no citation block at all (greeting / low confidence)

    return {
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": body,
        },
    }


def build_feedback_card(agent_response: dict) -> dict:
    """Separate small card with just 👍 👎 feedback actions."""
    question_id     = agent_response.get("question_id")
    answer_id       = agent_response.get("answer_id")
    conversation_id = agent_response.get("conversation_id")
    user_id         = agent_response.get("user_id")
    domain          = agent_response.get("domain") or "General"

    _fb = {
        "question_id":     question_id,
        "answer_id":       answer_id,
        "conversation_id": conversation_id,
        "user_id":         user_id,
        "domain":          domain,
    }

    def _show_card(feedback_type: str) -> dict:
        placeholder = "Add a comment (optional)" if feedback_type == "positive" \
                      else "What could be improved? (optional)"
        return {
            "type": "AdaptiveCard",
            "body": [{
                "type": "Input.Text", "id": "feedback_comment",
                "placeholder": placeholder, "isMultiline": True, "maxLength": 500,
            }],
            "actions": [{
                "type": "Action.Submit", "title": "Submit",
                "data": {**_fb, "action": "feedback", "feedback": feedback_type},
            }]
        }

    return {
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [{
                "type": "TextBlock", "text": "Was this helpful?",
                "size": "Small", "isSubtle": True, "spacing": "None",
            }],
            "actions": [
                {"type": "Action.ShowCard", "title": "👍", "card": _show_card("positive")},
                {"type": "Action.ShowCard", "title": "👎", "card": _show_card("negative")},
            ],
        },
    }
