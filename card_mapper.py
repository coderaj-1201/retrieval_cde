"""
card_mapper.py
Two separate cards:
1. build_answer_card  — answer + sources only, no buttons
2. build_feedback_card — just 👍 👎 with collapsible comment
"""
from __future__ import annotations
import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote as _url_quote, urlparse

logger = logging.getLogger(__name__)

# Approved URL scheme and hostname patterns for source links.
# Only URLs matching these are rendered as clickable links — anything else
# is dropped and shown as title-only to prevent phishing via KB documents.
_ALLOWED_SCHEMES  = frozenset({"https"})
_ALLOWED_HOST_RE  = re.compile(
    r"""
    ^(
        [\w-]+\.sharepoint\.com        |   # SharePoint / OneDrive
        [\w-]+\.blob\.core\.windows\.net|  # Azure Blob
        [\w-]+\.azurewebsites\.net     |   # Azure Web Apps
        [\w-]+\.microsoft\.com         |   # Microsoft docs
        [\w-]+\.ironman\.com               # IRONMAN domain
    )$
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _safe_url(url: str | None) -> str | None:
    """Validate URL against the approved domain allowlist, then percent-encode.

    Returns None if the URL is missing, uses a non-HTTPS scheme, or points to
    a host not on the allowlist — callers render title-only in that case.
    """
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except Exception:
        logger.warning("card_mapper_url_parse_error url=%.120s", url)
        return None
    if parsed.scheme not in _ALLOWED_SCHEMES:
        logger.warning("card_mapper_url_blocked scheme=%s url=%.120s", parsed.scheme, url)
        return None
    if not _ALLOWED_HOST_RE.match(parsed.netloc):
        logger.warning("card_mapper_url_blocked host=%s url=%.120s", parsed.netloc, url)
        return None
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


def _confidence_badge(score: float) -> tuple[str, str]:
    """Return (emoji+pct, description) for a citation confidence score."""
    pct = int(round(score * 100))
    if pct >= 80:
        return f"🟢 {pct}%", "High relevance"
    if pct >= 60:
        return f"🟡 {pct}%", "Moderate relevance"
    return f"🔴 {pct}%", "Low relevance"


def _citation_row(link_text: str, score: float | None = None) -> dict:
    """ColumnSet row: bullet + link on left, confidence badge right-aligned."""
    left_col: dict = {
        "type": "Column",
        "width": "stretch",
        "items": [{
            "type": "TextBlock",
            "text": f"• {link_text}",
            "wrap": True, "size": "Small",
        }],
    }
    if score is not None:
        badge, label = _confidence_badge(score)
        right_col: dict = {
            "type": "Column",
            "width": "auto",
            "horizontalAlignment": "right",
            "items": [{
                "type": "TextBlock",
                "text": f"{badge} *{label}*",
                "wrap": False, "size": "Small", "isSubtle": True,
                "horizontalAlignment": "right",
            }],
        }
        columns = [left_col, right_col]
    else:
        columns = [left_col]

    return {
        "type": "ColumnSet",
        "spacing": "Small",
        "columns": columns,
    }


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

    sources = normalize_sources(agent_response.get("sources", []))
    # Build title → url lookup so citations can be rendered as clickable links.
    url_map: dict[str, str | None] = {s["title"]: s.get("url") for s in sources}

    if show_citations and llm_citations:
        body.append({
            "type": "TextBlock", "text": "**Sources**",
            "wrap": True, "size": "Small", "weight": "Bolder",
            "spacing": "Medium", "separator": True,
        })
        for cite in llm_citations[:5]:
            title = cite.get("title") or "Source"
            score = float(cite.get("confidence", 0.0))
            url   = url_map.get(title)
            link  = f"[{title}]({url})" if url else title
            body.append(_citation_row(link, score))

    elif show_citations and not llm_citations:
        # LLM said show citations but returned none — fall back to search sources
        if sources:
            body.append({
                "type": "TextBlock", "text": "**Sources**",
                "wrap": True, "size": "Small", "weight": "Bolder",
                "spacing": "Medium", "separator": True,
            })
            for src in sources[:5]:
                title = src["title"]
                url   = src.get("url")
                link  = f"[{title}]({url})" if url else title
                body.append(_citation_row(link))

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
