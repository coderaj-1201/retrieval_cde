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
import re
from collections import OrderedDict
from datetime import datetime, timezone

from shared.config import settings
from shared.cosmos_client import (
    get_ltm_container, get_sessions_container,
    get_document, upsert_document, query_documents,
)
from shared.models import ConversationTurn, LongTermMemoryRecord, SessionMemory

logger = logging.getLogger(__name__)


# ── Async-safe LRU cache for session memory ───────────────────────────────────
# The plain OrderedDict implementation was not safe for concurrent asyncio
# coroutines — two coroutines racing on move_to_end / popitem could raise
# RuntimeError or silently corrupt the dict. This class serialises all
# access through an asyncio.Lock.

class _SessionLRUCache:
    def __init__(self, max_size: int = 200) -> None:
        self._cache: OrderedDict[str, SessionMemory] = OrderedDict()
        self._max   = max_size
        self._lock  = asyncio.Lock()

    async def get(self, key: str) -> SessionMemory | None:
        async with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    async def set(self, key: str, value: SessionMemory) -> None:
        async with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)


_session_cache = _SessionLRUCache(max_size=200)


# ── Short-term memory ─────────────────────────────────────────────────────────

async def load_session(conversation_id: str, user_id: str) -> SessionMemory:
    """Load session from cache → Cosmos → create new."""
    cached = await _session_cache.get(conversation_id)
    if cached:
        return cached

    try:
        doc = await asyncio.to_thread(
            get_document, get_sessions_container(), conversation_id, conversation_id
        )
    except Exception as exc:
        logger.error("session_load_failed conversation_id=%s — starting fresh: %s", conversation_id, exc, exc_info=True)
        doc = None

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

    await _session_cache.set(conversation_id, session)
    return session


async def append_turn(session: SessionMemory, turn: ConversationTurn) -> None:
    """Append turn, trim to window, persist to Cosmos."""
    session.turns.append(turn)
    if len(session.turns) > settings.SESSION_MAX_TURNS:
        session.turns = session.turns[-settings.SESSION_MAX_TURNS:]
    session.updated_at = datetime.now(timezone.utc).isoformat()
    await _session_cache.set(session.conversation_id, session)
    try:
        await asyncio.to_thread(upsert_document, get_sessions_container(), session.to_dict())
    except Exception as exc:
        logger.error("session_persist_failed conversation_id=%s: %s", session.conversation_id, exc, exc_info=True)
    logger.debug(
        "session_updated conversation_id=%s turns=%d",
        session.conversation_id, len(session.turns),
    )


# Phrases that strongly suggest the user is continuing a prior thread.
_FOLLOWUP_PHRASES = (
    "what about", "and the", "tell me more", "more details", "what else",
    "how about", "can you explain", "elaborate", "and what", "what if",
    "same for", "related to", "similar to", "what is the difference",
)
# Bare pronouns that reference something from a prior turn.
_PRONOUN_RE = re.compile(
    r'\b(it|that|this|they|their|its|those|these|them|he|she|we|same)\b',
    re.IGNORECASE,
)
# Max chars of answer shown per turn in the context block.
_SESSION_CONTEXT_ANSWER_CHARS = 150
# Always include the last 3 turns — kept small to stay lean on tokens.
# The classifier LLM now decides follow-up detection via is_followup;
# we always provide a small window so it has enough context without
# bloating every prompt.
_SESSION_CONTEXT_TURNS = 3


def needs_session_context(query: str) -> bool:
    """
    Heuristic retained as a fallback for the rewrite step.
    Primary follow-up detection is now done by the classifier LLM via
    the is_followup field — this function is no longer the main gate.
    """
    q = query.strip().lower()
    if len(q) < 40:
        return True
    if any(phrase in q for phrase in _FOLLOWUP_PHRASES):
        return True
    if _PRONOUN_RE.search(q):
        return True
    return False


def format_session_context(session: SessionMemory, query: str = "") -> str:
    """
    Always returns the last few turns so the classifier LLM can decide
    whether the query is a follow-up. The old rule-based gate caused it
    to miss topic-reference follow-ups like "What about the approval
    process?" that lack explicit pronouns.
    """
    if not session.turns:
        return ""
    recent = session.turns[-_SESSION_CONTEXT_TURNS:]
    lines = [f"## Recent conversation (last {len(recent)} turn(s))"]
    for t in recent:
        lines.append(f"Q: {t.question}")
        answer_excerpt = t.answer[:_SESSION_CONTEXT_ANSWER_CHARS]
        if len(t.answer) > _SESSION_CONTEXT_ANSWER_CHARS:
            answer_excerpt += "…"
        lines.append(f"A: {answer_excerpt}")
    return "\n".join(lines)


# ── Long-term memory ──────────────────────────────────────────────────────────

async def load_ltm(user_id: str) -> LongTermMemoryRecord | None:
    try:
        doc = await asyncio.to_thread(
            get_document, get_ltm_container(), f"ltm-{user_id}", user_id
        )
    except Exception as exc:
        logger.error("ltm_load_failed user_id=%s: %s", user_id, exc, exc_info=True)
        return None
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
