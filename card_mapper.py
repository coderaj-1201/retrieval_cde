"""
card_mapper.py
Two separate cards:
1. build_answer_card  — answer + sources only, no buttons
2. build_feedback_card — just 👍 👎 with collapsible comment
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any


def normalize_sources(sources: Any) -> list[dict]:
    if not isinstance(sources, list):
        return []
    seen = set()
    result = []
    for i, s in enumerate(sources, start=1):
        if not isinstance(s, dict):
            continue
        title = (
            s.get("title") or s.get("name") or s.get("document_name")
            or s.get("documentName") or f"Source {i}"
        )
        if title in seen:
            continue
        seen.add(title)
        result.append({
            "title": title,
            "url":   s.get("url") or s.get("source_url") or s.get("sourceUrl"),
            "page":  s.get("page") or s.get("page_number"),
        })
    return result


def build_answer_card(agent_response: dict) -> dict:
    """Answer + sources only. No feedback buttons."""
    answer  = (agent_response.get("answer") or "").strip()
    sources = normalize_sources(agent_response.get("sources", []))

    body: list[dict] = [
        {"type": "TextBlock", "text": answer, "wrap": True, "spacing": "None"},
    ]

    if sources:
        body.append({
            "type": "TextBlock", "text": "**Sources**",
            "wrap": True, "size": "Small", "weight": "Bolder",
            "spacing": "Medium", "separator": True,
        })
        for i, src in enumerate(sources[:5], start=1):
            page = f" · Page {src['page']}" if src.get("page") else ""
            title = src["title"]
            url   = src.get("url")
            line  = f"[{i}. {title}]({url}){page}" if url else f"{i}. {title}{page}"
            body.append({
                "type": "TextBlock", "text": line,
                "wrap": True, "size": "Small", "isSubtle": True, "spacing": "Small",
            })

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
