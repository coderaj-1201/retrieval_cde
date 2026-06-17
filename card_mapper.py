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


def build_escalation_card(agent_response: dict, question_text: str = "") -> dict:
    """Card shown when the bot couldn't answer confidently — lets the user
    manually trigger a Zendesk ticket or SME connection via button click.
    No ticket is ever raised automatically; this card is the only trigger.
    """
    answer = (agent_response.get("answer") or "").strip()
    opts   = agent_response.get("escalation_options") or {}

    _ctx = {
        "question_id":     agent_response.get("question_id"),
        "answer_id":        agent_response.get("answer_id"),
        "conversation_id":  agent_response.get("conversation_id"),
        "user_id":          agent_response.get("user_id"),
        "domain":           agent_response.get("domain") or "",
        "question_text":    question_text,
    }

    body: list[dict] = []
    if answer:
        body.append({"type": "TextBlock", "text": answer, "wrap": True, "spacing": "None"})
    body.append({
        "type": "TextBlock",
        "text": "Need more help? Choose an option below:",
        "wrap": True, "size": "Small", "isSubtle": True, "spacing": "Medium",
    })

    actions: list[dict] = []
    if "raise_ticket" in opts:
        actions.append({
            "type": "Action.Submit",
            "title": f"🎫 Raise Ticket ({opts['raise_ticket'].get('sla', '')})",
            "data": {**_ctx, "action": "escalate", "escalation_type": "raise_ticket"},
        })
    if "connect_sme" in opts:
        actions.append({
            "type": "Action.Submit",
            "title": f"🙋 Connect SME ({opts['connect_sme'].get('sla', '')})",
            "data": {**_ctx, "action": "escalate", "escalation_type": "connect_sme"},
        })

    return {
        "contentType": "application/vnd.microsoft.card.adaptive",
        "content": {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": body,
            "actions": actions,
        },
    }


def build_escalation_confirmation_card(data: dict) -> dict:
    """Confirmation card shown after the user clicks Raise Ticket or Connect SME.

    Displays the escalation type, reference ID, SLA, and a copy-reference button.
    Works for both ticket_raised and sme_connecting statuses.
    """
    status         = data.get("status", "ticket_raised")
    correlation_id = data.get("correlation_id", "")
    domain         = (data.get("domain") or "").upper()
    answer         = (data.get("answer") or "").strip()

    is_ticket = status == "ticket_raised"
    icon      = "🎫" if is_ticket else "🙋"
    title     = "Support Ticket Raised" if is_ticket else "SME Connection Requested"
    sla_label = "Expected response time" if is_ticket else "Expected callback time"
    sla_value = "4 business hours" if is_ticket else "2 business hours"
    color     = "Good"  # green header bar

    body: list[dict] = [
        {
            "type": "ColumnSet",
            "columns": [
                {
                    "type": "Column", "width": "auto",
                    "items": [{"type": "TextBlock", "text": icon, "size": "ExtraLarge", "spacing": "None"}],
                },
                {
                    "type": "Column", "width": "stretch",
                    "items": [
                        {
                            "type": "TextBlock",
                            "text": title,
                            "weight": "Bolder", "size": "Medium", "spacing": "None",
                            "color": color,
                        },
                        *([{
                            "type": "TextBlock",
                            "text": f"Domain: {domain}",
                            "size": "Small", "isSubtle": True, "spacing": "None",
                        }] if domain else []),
                    ],
                },
            ],
        },
        {"type": "Separator"},
    ]

    if correlation_id:
        body += [
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Reference", "value": correlation_id},
                    {"title": sla_label,   "value": sla_value},
                ],
                "spacing": "Small",
            },
        ]
    elif answer:
        body.append({
            "type": "TextBlock", "text": answer,
            "wrap": True, "size": "Small", "spacing": "Small",
        })

    body.append({
        "type": "TextBlock",
        "text": "Our support team will reach out to you shortly.",
        "wrap": True, "size": "Small", "isSubtle": True, "spacing": "Medium",
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
