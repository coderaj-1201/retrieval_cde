"""
Memory manager — short-term (session) + long-term (per-user summary).

Short-term  : last SESSION_MAX_TURNS turns per conversation_id.
              Stored in Cosmos `sessions` container.
              Also kept in process-local LRU cache for low-latency reads.

Long-term   : rolling LLM-generated summary + extracted key_facts per user_id.
              Stored in Cosmos `long-term-memory` container.
              Updated every LTM_SUMMARY_EVERY_N turns.

Both are injected into the orchestrator's classify_query system prompt so the
LLM has full context when routing and synthesising answers.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from datetime import datetime, timezone

from shared.config import settings
from shared.cosmos_client import (
    get_ltm_container, get_sessions_container,
    get_document, upsert_document, query_documents,
)
from shared.models import ConversationTurn, LongTermMemoryRecord, SessionMemory

logger = logging.getLogger(__name__)

# ── Process-local LRU cache for session memory (avoids Cosmos round-trip) ─────
_SESSION_CACHE: OrderedDict[str, SessionMemory] = OrderedDict()
_SESSION_CACHE_MAX = 200


def _cache_set(conv_id: str, session: SessionMemory) -> None:
    _SESSION_CACHE[conv_id] = session
    _SESSION_CACHE.move_to_end(conv_id)
    if len(_SESSION_CACHE) > _SESSION_CACHE_MAX:
        _SESSION_CACHE.popitem(last=False)


def _cache_get(conv_id: str) -> SessionMemory | None:
    if conv_id in _SESSION_CACHE:
        _SESSION_CACHE.move_to_end(conv_id)
        return _SESSION_CACHE[conv_id]
    return None


# ── Short-term memory ─────────────────────────────────────────────────────────

async def load_session(conversation_id: str, user_id: str) -> SessionMemory:
    """Load session from cache → Cosmos → create new."""
    cached = _cache_get(conversation_id)
    if cached:
        return cached

    doc = await asyncio.to_thread(
        get_document, get_sessions_container(), conversation_id, conversation_id
    )
    if doc:
        turns = [ConversationTurn(**t) for t in doc.get("turns", [])]
        session = SessionMemory(
            conversation_id=conversation_id,
            user_id=user_id,
            turns=turns,
            created_at=doc.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=doc.get("updated_at", datetime.now(timezone.utc).isoformat()),
        )
    else:
        session = SessionMemory(conversation_id=conversation_id, user_id=user_id)

    _cache_set(conversation_id, session)
    return session


async def append_turn(session: SessionMemory, turn: ConversationTurn) -> None:
    """Append turn, trim to window, persist to Cosmos."""
    session.turns.append(turn)
    if len(session.turns) > settings.SESSION_MAX_TURNS:
        session.turns = session.turns[-settings.SESSION_MAX_TURNS:]
    session.updated_at = datetime.now(timezone.utc).isoformat()
    _cache_set(session.conversation_id, session)
    await asyncio.to_thread(upsert_document, get_sessions_container(), session.to_dict())
    logger.debug("session updated conversation_id=%s turns=%d", session.conversation_id, len(session.turns))


def format_session_context(session: SessionMemory) -> str:
    """Render recent turns as a compact string for prompt injection."""
    if not session.turns:
        return ""
    lines = ["## Recent conversation history"]
    for t in session.turns[-5:]:   # last 5 turns max in prompt
        lines.append(f"Q: {t.question}")
        lines.append(f"A: {t.answer[:300]}{'...' if len(t.answer) > 300 else ''}")
    return "\n".join(lines)


# ── Long-term memory ──────────────────────────────────────────────────────────

async def load_ltm(user_id: str) -> LongTermMemoryRecord | None:
    doc = await asyncio.to_thread(
        get_document, get_ltm_container(), f"ltm-{user_id}", user_id
    )
    if doc:
        return LongTermMemoryRecord(
            id=doc["id"],
            user_id=doc["user_id"],
            summary=doc.get("summary", ""),
            key_facts=doc.get("key_facts", []),
            last_updated=doc.get("last_updated", ""),
            source_conversation_ids=doc.get("source_conversation_ids", []),
        )
    return None


async def update_ltm(user_id: str, session: SessionMemory) -> None:
    """
    Called every LTM_SUMMARY_EVERY_N turns. Uses LLM to produce a rolling
    summary + key facts list from the full session history.
    """
    from shared.azure_clients import get_openai_client  # noqa: PLC0415

    existing = await load_ltm(user_id)
    prior_summary = existing.summary if existing else ""
    prior_facts   = existing.key_facts if existing else []

    # Bound the prior summary and facts to avoid token overflow on long-lived users.
    prior_summary_bounded = prior_summary[:settings.LTM_MAX_SUMMARY_CHARS]
    prior_facts_bounded   = prior_facts[:settings.LTM_MAX_FACTS]

    if len(prior_summary) > settings.LTM_MAX_SUMMARY_CHARS:
        logger.warning(
            "ltm_summary_truncated user_id=%s original_len=%d bounded_len=%d",
            user_id, len(prior_summary), settings.LTM_MAX_SUMMARY_CHARS,
        )

    all_text = "\n".join(
        f"Q: {t.question}\nA: {t.answer}" for t in session.turns
    )

    system = (
        "You are a memory assistant. Given prior summary, prior facts, and new conversation turns, "
        "produce an updated summary (max 150 words) and an updated list of key facts (max 15 bullet strings). "
        "Return ONLY JSON: {\"summary\": \"...\", \"key_facts\": [\"...\", ...]}"
    )
    user_msg = (
        f"Prior summary:\n{prior_summary_bounded}\n\n"
        f"Prior key facts:\n{json.dumps(prior_facts_bounded)}\n\n"
        f"New turns:\n{all_text}"
    )

    try:
        resp = await asyncio.to_thread(
            get_openai_client().chat.completions.create,
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content)
        summary    = raw.get("summary", prior_summary_bounded)
        key_facts  = raw.get("key_facts", prior_facts_bounded)
    except Exception as exc:
        logger.error(
            "ltm_update_llm_failed user_id=%s: %s",
            user_id, exc, exc_info=True,
        )
        return

    src_ids = list({*(existing.source_conversation_ids if existing else []), session.conversation_id})
    record = LongTermMemoryRecord(
        id=f"ltm-{user_id}",
        user_id=user_id,
        summary=summary,
        key_facts=key_facts,
        source_conversation_ids=src_ids,
    )
    await asyncio.to_thread(upsert_document, get_ltm_container(), record.to_dict())
    logger.info("LTM updated user_id=%s facts=%d", user_id, len(key_facts))


def format_ltm_context(ltm: LongTermMemoryRecord | None) -> str:
    """Render LTM as a compact string for prompt injection."""
    if not ltm or not ltm.summary:
        return ""
    lines = ["## Long-term user context"]
    lines.append(ltm.summary)
    if ltm.key_facts:
        lines.append("Key facts:")
        lines.extend(f"- {f}" for f in ltm.key_facts[:10])
    return "\n".join(lines)
