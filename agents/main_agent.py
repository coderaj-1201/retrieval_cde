"""
Main Agent
==========
Entry point for all queries. Calls Orchestrator via HTTP.

Endpoints:
  POST /query          — main RAG query
  POST /feedback       — submit thumbs-up/down + comment
  GET  /feedback       — retrieve feedback by answer_id or user_id
  GET  /chat-history   — retrieve conversation turns for a user/conversation
  GET  /health         — deep health check (Cosmos + OpenAI probe)
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from agent_framework import step, workflow
from fastapi import FastAPI, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from shared.cosmos_client import (
    get_chat_container, get_feedback_container,
    probe_cosmos, upsert_document, query_documents, get_document,
)
from shared.logging_config import bind_context, configure_logging, get_logger
from shared.memory import (
    append_turn, format_ltm_context, format_session_context,
    load_ltm, load_session, update_ltm,
)
from shared.models import (
    ChatHistoryRecord, ConversationTurn, Domain, FeedbackRating,
    FeedbackRecord, FinalResponse, OrchestratorInput, QueryResponse, UserQuery,
)
from shared.config import settings
from shared.rate_limiter import RateLimitExceeded, check_rate_limit
import os
from dotenv import load_dotenv
load_dotenv()

configure_logging()
logger = get_logger(__name__)

_ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL")

_ESCALATION_OPTIONS = {
    "raise_ticket": {
        "action":      "raise_ticket",
        "description": "Raise a support ticket",
        "reply_with":  "raise_ticket",
        "sla":         "4 business hours",
    },
    "connect_sme": {
        "action":      "connect_sme",
        "description": "Connect with a Subject Matter Expert",
        "reply_with":  "connect_sme",
        "sla":         "2 business hours",
    },
}


# ── Pydantic request bodies ────────────────────────────────────────────────────

class QueryBody(BaseModel):
    text: str
    conversation_id: str | None = None
    user_id: str                = "anonymous"
    idempotency_key: str | None = None   # optional — prevents duplicate processing on retry


class FeedbackBody(BaseModel):
    question_id: str
    answer_id: str
    conversation_id: str
    user_id: str          = "anonymous"
    rating: FeedbackRating
    comment: str          = ""


# ── Workflow steps ─────────────────────────────────────────────────────────────

@step
async def call_orchestrator(inp: OrchestratorInput) -> FinalResponse:
    user_query      = inp.user_query
    session_context = inp.session_context
    ltm_context     = inp.ltm_context
    payload = {
        "text":            user_query.text,
        "conversation_id": user_query.conversation_id,
        "user_id":         user_query.user_id,
        "question_id":     user_query.question_id,
        "session_context": session_context,
        "ltm_context":     ltm_context,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{_ORCHESTRATOR_URL}/orchestrate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        domain_val = data.get("domain") or ""
        try:
            domain = Domain(domain_val.lower()) if domain_val else None
        except ValueError:
            domain = None
        return FinalResponse(
            status=data.get("status", "failure"),
            answer=data.get("answer", ""),
            domain=domain,
            sources=data.get("sources", []),
            confidence=float(data.get("confidence", 0.0)),
            attempts_used=int(data.get("attempts_used", 0)),
            conversation_id=data.get("conversation_id", user_query.conversation_id),
            user_id=data.get("user_id", user_query.user_id),
            question_id=data.get("question_id", user_query.question_id),
            answer_id=data.get("answer_id", f"ans-{uuid.uuid4().hex[:12]}"),
            tools_used=data.get("tools_used", []),
        )


@step
async def handle_raise_ticket(user_id: str, conversation_id: str) -> QueryResponse:
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    logger.info("ticket_raised ticket_id=%s", ticket_id)
    return QueryResponse(
        question_id=f"q-{uuid.uuid4().hex[:12]}",
        answer_id=f"ans-{uuid.uuid4().hex[:12]}",
        conversation_id=conversation_id,
        user_id=user_id,
        status="ticket_raised",
        answer=f"✅ Ticket raised. Reference: `{ticket_id}`. Expected response: 4 business hours.",
        domain="",
        confidence=1.0,
        attempts_used=0,
        tools_used=[],
        sources=[],
        escalation_options=None,
    )


@step
async def handle_connect_sme(user_id: str, conversation_id: str) -> QueryResponse:
    logger.info("sme_connect_requested user_id=%s", user_id)
    return QueryResponse(
        question_id=f"q-{uuid.uuid4().hex[:12]}",
        answer_id=f"ans-{uuid.uuid4().hex[:12]}",
        conversation_id=conversation_id,
        user_id=user_id,
        status="sme_connecting",
        answer="✅ Connecting you with an SME. Expected response: 2 business hours.",
        domain="",
        confidence=1.0,
        attempts_used=0,
        tools_used=[],
        sources=[],
        escalation_options=None,
    )


@workflow(name="main_agent_workflow")
async def main_agent_workflow(user_query: UserQuery) -> QueryResponse:
    text = user_query.text.strip().lower()

    if text == "raise_ticket":
        return await handle_raise_ticket(user_query.user_id, user_query.conversation_id)
    if text == "connect_sme":
        return await handle_connect_sme(user_query.user_id, user_query.conversation_id)

    session = await load_session(user_query.conversation_id, user_query.user_id)
    ltm     = await load_ltm(user_query.user_id)

    try:
        final: FinalResponse = await call_orchestrator(OrchestratorInput(
            user_query=user_query,
            session_context=format_session_context(session),
            ltm_context=format_ltm_context(ltm),
        ))
    except Exception as exc:
        logger.error("orchestrator_call_failed: %s", exc, exc_info=True)
        return QueryResponse(
            question_id=user_query.question_id,
            answer_id=f"ans-{uuid.uuid4().hex[:12]}",
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            status="error",
            answer="Service temporarily unavailable. Please try again.",
            domain="",
            confidence=0.0,
            attempts_used=0,
            tools_used=[],
            sources=[],
            escalation_options=_ESCALATION_OPTIONS,
        )

    is_success = final.status == "success"
    # B2 fix: use .value on StrEnum, not str() which produces "Domain.HR"
    domain_str = final.domain.value.upper() if isinstance(final.domain, Domain) else (final.domain or "")

    response = QueryResponse(
        question_id=user_query.question_id,
        answer_id=final.answer_id,
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        status=final.status,
        answer=final.answer if is_success else "",
        domain=domain_str,
        confidence=final.confidence,
        attempts_used=final.attempts_used,
        tools_used=final.tools_used,
        sources=final.sources,
        escalation_options=None if is_success else _ESCALATION_OPTIONS,
    )

    # Persist to Cosmos
    upsert_document(get_chat_container(), ChatHistoryRecord(
        id=user_query.question_id,
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        question_id=user_query.question_id,
        answer_id=final.answer_id,
        question=user_query.text,
        answer=final.answer,
        domain=domain_str,
        confidence=final.confidence,
        tools_used=final.tools_used,
        sources=final.sources,
        status=final.status,
    ).to_dict())

    # Short-term memory
    await append_turn(session, ConversationTurn(
        question_id=user_query.question_id,
        answer_id=final.answer_id,
        question=user_query.text,
        answer=final.answer,
        domain=domain_str,
        confidence=final.confidence,
        tools_used=final.tools_used,
    ))

    # Long-term memory — background, never blocks response
    if len(session.turns) % settings.LTM_SUMMARY_EVERY_N == 0:
        asyncio.create_task(_run_ltm_update(user_query.user_id, session))

    return response


async def _run_ltm_update(user_id: str, session) -> None:
    try:
        await update_ltm(user_id, session)
    except Exception as exc:
        logger.error("ltm_update_failed user_id=%s: %s", user_id, exc)


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # B1 fix: probe_cosmos is blocking — run in thread pool, not event loop
    await asyncio.to_thread(probe_cosmos)
    logger.info("main_agent_started")
    yield
    logger.info("main_agent_stopped")


app = FastAPI(title="RAG Main Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ask-ops-bot-frontend-e9dfe0aqgfdcg7e3.southcentralus-01.azurewebsites.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── F1: Deep health check ──────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Response:
    checks: dict[str, str] = {}
    overall_ok = True

    # Cosmos probe
    try:
        await asyncio.to_thread(get_chat_container().read)
        checks["cosmos"] = "ok"
    except Exception as exc:
        checks["cosmos"] = f"error: {exc}"
        overall_ok = False

    # OpenAI probe — cheapest possible call
    try:
        from shared.azure_clients import get_openai_client
        await asyncio.to_thread(
            get_openai_client().models.list
        )
        checks["openai"] = "ok"
    except Exception as exc:
        checks["openai"] = f"error: {exc}"
        overall_ok = False

    # Orchestrator probe
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{_ORCHESTRATOR_URL}/health")
            checks["orchestrator"] = "ok" if r.status_code == 200 else f"status={r.status_code}"
            if r.status_code != 200:
                overall_ok = False
    except Exception as exc:
        checks["orchestrator"] = f"error: {exc}"
        overall_ok = False

    http_status = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return Response(
        content=json.dumps({
            "status": "healthy" if overall_ok else "degraded",
            "agent":  "main",
            "checks": checks,
        }),
        media_type="application/json",
        status_code=http_status,
    )


# ── POST /query ────────────────────────────────────────────────────────────────

@app.post("/query")
async def query(body: QueryBody) -> Response:
    # F8: rate limit — returns 429 before any work is done
    try:
        check_rate_limit(body.user_id)
    except RateLimitExceeded as exc:
        logger.warning("rate_limit_exceeded user_id=%s retry_after=%.1f", body.user_id, exc.retry_after)
        return Response(
            content=json.dumps({
                "error":       "rate_limit_exceeded",
                "retry_after": exc.retry_after,
                "message":     f"Too many requests. Please wait {exc.retry_after}s before retrying.",
            }),
            media_type="application/json",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(exc.retry_after)},
        )

    conversation_id = body.conversation_id or str(uuid.uuid4())

    # F2: idempotency — if key already processed, return cached result
    if body.idempotency_key:
        cached = await asyncio.to_thread(
            get_document,
            get_chat_container(),
            body.idempotency_key,
            conversation_id,
        )
        if cached:
            logger.info("idempotency_hit key=%s", body.idempotency_key)
            clean = {k: v for k, v in cached.items() if not k.startswith("_")}
            return Response(
                content=json.dumps(clean),
                media_type="application/json",
                headers={"X-Idempotency": "hit"},
            )

    user_query = UserQuery(
        text=body.text,
        conversation_id=conversation_id,
        user_id=body.user_id,
        # Use idempotency_key as question_id when provided so the caller
        # can correlate retries to the same Cosmos record
        question_id=body.idempotency_key or f"q-{uuid.uuid4().hex[:12]}",
    )
    bind_context(
        agent="main",
        conversation_id=conversation_id,
        user_id=body.user_id,
        question_id=user_query.question_id,
    )
    logger.info("query_received text_preview=%.80s", body.text)

    result_obj = await main_agent_workflow.run(user_query)
    outputs    = result_obj.get_outputs()
    response: QueryResponse = outputs[0] if outputs else QueryResponse(
        question_id=user_query.question_id,
        answer_id="",
        conversation_id=conversation_id,
        user_id=body.user_id,
        status="error",
        answer="Internal error.",
        domain="",
        confidence=0.0,
        attempts_used=0,
        tools_used=[],
        sources=[],
        escalation_options=_ESCALATION_OPTIONS,
    )

    logger.info(
        "query_complete question_id=%s answer_id=%s status=%s confidence=%.3f",
        response.question_id, response.answer_id, response.status, response.confidence,
    )
    return Response(
        content=json.dumps(response.to_dict()),
        media_type="application/json",
    )


# ── POST /feedback ─────────────────────────────────────────────────────────────

@app.post("/feedback")
async def feedback_post(body: FeedbackBody) -> Response:
    bind_context(
        agent="main",
        conversation_id=body.conversation_id,
        user_id=body.user_id,
        question_id=body.question_id,
    )
    logger.info(
        "feedback_received question_id=%s answer_id=%s rating=%s",
        body.question_id, body.answer_id, body.rating,
    )
    record = FeedbackRecord(
        id=f"fb-{uuid.uuid4().hex[:12]}",
        question_id=body.question_id,
        answer_id=body.answer_id,
        user_id=body.user_id,
        conversation_id=body.conversation_id,
        rating=body.rating,
        comment=body.comment,
    )
    upsert_document(get_feedback_container(), record.to_dict())
    return Response(
        content=json.dumps({
            "status":      "ok",
            "feedback_id": record.id,
            "question_id": body.question_id,
            "answer_id":   body.answer_id,
            "rating":      body.rating,
            "timestamp":   record.timestamp,
        }),
        media_type="application/json",
    )


# ── F3: GET /feedback ──────────────────────────────────────────────────────────

@app.get("/feedback")
async def feedback_get(
    answer_id:       str | None = Query(default=None),
    question_id:     str | None = Query(default=None),
    conversation_id: str | None = Query(default=None),
    user_id:         str        = Query(default="anonymous"),
    limit:           int        = Query(default=20, ge=1, le=100),
) -> Response:
    bind_context(agent="main", user_id=user_id)

    if answer_id:
        cosmos_query = (
            "SELECT * FROM c WHERE c.answer_id = @answer_id AND c.type = 'feedback' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@answer_id", "value": answer_id},
            {"name": "@limit",     "value": limit},
        ]
    elif question_id:
        cosmos_query = (
            "SELECT * FROM c WHERE c.question_id = @question_id AND c.type = 'feedback' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@question_id", "value": question_id},
            {"name": "@limit",       "value": limit},
        ]
    elif conversation_id:
        cosmos_query = (
            "SELECT * FROM c WHERE c.conversation_id = @conv_id AND c.type = 'feedback' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@conv_id", "value": conversation_id},
            {"name": "@limit",   "value": limit},
        ]
    else:
        cosmos_query = (
            "SELECT * FROM c WHERE c.user_id = @user_id AND c.type = 'feedback' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@user_id", "value": user_id},
            {"name": "@limit",   "value": limit},
        ]

    docs = query_documents(get_feedback_container(), cosmos_query, params)
    clean = [{k: v for k, v in d.items() if not k.startswith("_")} for d in docs]

    # Aggregate rating summary when querying by answer_id or question_id
    summary: dict | None = None
    if answer_id or question_id:
        counts: dict[str, int] = {}
        for d in clean:
            r = d.get("rating", "")
            counts[r] = counts.get(r, 0) + 1
        summary = {"total": len(clean), "by_rating": counts}

    return Response(
        content=json.dumps({"count": len(clean), "summary": summary, "feedback": clean}),
        media_type="application/json",
    )


# ── GET /chat-history ──────────────────────────────────────────────────────────

@app.get("/chat-history")
async def chat_history(
    conversation_id: str | None = Query(default=None),
    user_id:         str        = Query(default="anonymous"),
    limit:           int        = Query(default=20, ge=1, le=100),
) -> Response:
    bind_context(agent="main", conversation_id=conversation_id or "", user_id=user_id)
    logger.info("chat_history_requested conversation_id=%s user_id=%s", conversation_id, user_id)

    if conversation_id:
        cosmos_query = (
            "SELECT * FROM c WHERE c.conversation_id = @conv_id AND c.type = 'chat_history' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@conv_id", "value": conversation_id},
            {"name": "@limit",   "value": limit},
        ]
    else:
        cosmos_query = (
            "SELECT * FROM c WHERE c.user_id = @user_id AND c.type = 'chat_history' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@user_id", "value": user_id},
            {"name": "@limit",   "value": limit},
        ]

    docs  = query_documents(get_chat_container(), cosmos_query, params)
    clean = [{k: v for k, v in d.items() if not k.startswith("_")} for d in docs]
    return Response(
        content=json.dumps({"count": len(clean), "history": clean}),
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run("agents.main_agent:app", host="0.0.0.0", port=8000, reload=False)
