"""
Main Agent
==========
Entry point for all queries. Calls Orchestrator via HTTP.

Endpoints:
  POST /query          — main RAG query
  POST /feedback       — submit thumbs-up/down + comment
  GET  /feedback       — retrieve feedback (scoped by user_id or conversation_id)
  GET  /chat-history   — retrieve conversation turns (scoped by conversation_id)
  GET  /health         — liveness + readiness checks
"""
from __future__ import annotations

import asyncio
import html
import json
import re
import uuid
from contextlib import asynccontextmanager

import httpx
import signal
import uvicorn
from agent_framework import step, workflow
from fastapi import FastAPI, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from shared.circuit_breaker import CircuitBreaker, CircuitOpenError
from shared.config import settings
from shared.cosmos_client import (
    get_chat_container, get_feedback_container,
    probe_cosmos, upsert_document, query_documents, get_document,
)
from shared.escalation_client import (
    connect_sme as sb_connect_sme,
    is_escalation_configured,
    raise_ticket as sb_raise_ticket,
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
from shared.rate_limiter import RateLimitExceeded, check_rate_limit
from shared.telemetry import record_attempts, record_confidence, record_escalation, record_query
import os
from dotenv import load_dotenv
load_dotenv()

configure_logging()
logger = get_logger(__name__)

_ORCHESTRATOR_URL      = os.getenv("ORCHESTRATOR_URL")
_http: httpx.AsyncClient | None = None          # shared client — set in lifespan
_orchestrator_breaker  = CircuitBreaker(name="orchestrator-agent", fail_max=3, reset_timeout=30)


def _internal_headers() -> dict[str, str]:
    secret = (
        settings.INTERNAL_API_SECRET.get_secret_value()
        if settings.INTERNAL_API_SECRET is not None
        else None
    )
    return {"X-Internal-Secret": secret} if secret else {}

_ESCALATION_OPTIONS = {
    "raise_ticket": {
        "action":      "raise_ticket",
        "description": "Raise a support ticket",
        "reply_with":  "raise_ticket",
        "sla":         settings.ESCALATION_SLA_TICKET,
    },
    "connect_sme": {
        "action":      "connect_sme",
        "description": "Connect with a Subject Matter Expert",
        "reply_with":  "connect_sme",
        "sla":         settings.ESCALATION_SLA_SME,
    },
}

# Patterns that signal likely prompt injection attempts — logged as WARNING
# but not rejected outright (LLM guardrails handle the actual defence).
_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?(previous|above|prior)\s+instructions"
    r"|disregard\s+(all\s+)?instructions"
    r"|you\s+are\s+now\s+"
    r"|system\s+prompt"
    r"|jailbreak)",
    re.IGNORECASE,
)


# ── Payload size limit middleware ──────────────────────────────────────────────

class _ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with a body larger than 1 MB before they are read."""
    _MAX_BYTES = 1_048_576  # 1 MB

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self._MAX_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": "payload_too_large", "max_bytes": self._MAX_BYTES},
            )
        return await call_next(request)


# ── Pydantic request bodies ────────────────────────────────────────────────────

class QueryBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=settings.MAX_QUERY_LENGTH)
    conversation_id: str | None = None
    user_id: str                = "anonymous"
    idempotency_key: str | None = None

    @field_validator("text")
    @classmethod
    def sanitise_text(cls, v: str) -> str:
        # Strip control characters (null bytes, BEL, BS, etc.) but keep
        # printable unicode, tabs, and newlines.
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v).strip()
        if not v:
            raise ValueError("Query text is empty after sanitisation.")
        return v


class FeedbackBody(BaseModel):
    question_id:     str = Field(min_length=1, max_length=128)
    answer_id:       str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=256)
    user_id:         str = Field(default="anonymous", max_length=256)
    rating: FeedbackRating
    comment: str          = Field(default="", max_length=2000)

    @field_validator("comment")
    @classmethod
    def escape_html(cls, v: str) -> str:
        """HTML-escape comment before storage to prevent XSS in dashboards."""
        return html.escape(v, quote=True)


# ── Workflow steps ─────────────────────────────────────────────────────────────

async def _do_orchestrate(payload: dict) -> dict:
    """Raw HTTP call to the orchestrator — wrapped by circuit breaker."""
    global _http
    client = _http or httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=120.0))
    # X-Request-ID threads the question_id through all agents for log correlation.
    headers = {**_internal_headers(), "X-Request-ID": payload.get("question_id", "")}
    resp = await client.post(
        f"{_ORCHESTRATOR_URL}/orchestrate",
        json=payload,
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


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
    try:
        data = await _orchestrator_breaker.call(_do_orchestrate, payload)
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
            show_citations=bool(data.get("show_citations", False)),
            citations=data.get("citations", []),
        )
    except CircuitOpenError as exc:
        logger.error(
            "orchestrator_circuit_open retry_after=%.1f question_id=%s",
            exc.retry_after, user_query.question_id,
        )
        raise


@step
async def handle_raise_ticket(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
) -> QueryResponse:
    # Idempotency: one ticket per (user, conversation). If a ticket was already
    # raised for this conversation, return the existing reference instead of
    # creating a duplicate.
    ticket_idem_id = f"ticket-{user_id}-{conversation_id}"
    existing = await asyncio.to_thread(
        get_document, get_chat_container(), ticket_idem_id, conversation_id
    )
    if existing:
        existing_ref = existing.get("correlation_id", ticket_idem_id)
        logger.info("ticket_duplicate_suppressed ref=%s user_id=%s", existing_ref, user_id)
        return QueryResponse(
            question_id=question_id,
            answer_id=f"ans-{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            user_id=user_id,
            status="ticket_raised",
            answer=(
                f"A ticket was already raised for this conversation. "
                f"Reference: `{existing_ref}`. No duplicate created."
            ),
            domain=domain,
            confidence=1.0,
            attempts_used=0,
            tools_used=[],
            sources=[],
            escalation_options=None,
        )
    if is_escalation_configured():
        try:
            correlation_id = await asyncio.to_thread(
                sb_raise_ticket,
                user_id, conversation_id, question_id, question_text, domain,
            )
            # Persist idempotency record so duplicate clicks are suppressed.
            await asyncio.to_thread(
                upsert_document, get_chat_container(),
                {"id": ticket_idem_id, "correlation_id": correlation_id,
                 "conversation_id": conversation_id, "user_id": user_id},
            )
            answer = (
                f"Your ticket has been raised. Reference: `{correlation_id}`. "
                f"Expected response within **{settings.ESCALATION_SLA_TICKET}**."
            )
            record_escalation(escalation_type="raise_ticket", domain=domain or "unknown")
            logger.info(
                "ticket_queued correlation_id=%s user_id=%s domain=%s",
                correlation_id, user_id, domain,
            )
        except Exception as exc:
            logger.error("ticket_queue_failed user_id=%s: %s", user_id, exc, exc_info=True)
            correlation_id = f"REF-PENDING-{uuid.uuid4().hex[:6].upper()}"
            answer = (
                "Your escalation has been received but could not be queued automatically. "
                "Please contact the support team directly. "
                f"Reference: `{correlation_id}`."
            )
    else:
        # Service Bus not configured — log clearly so this is never silently swallowed.
        logger.error(
            "ticket_queue_skipped: Service Bus not configured. "
            "Set AZURE_SERVICE_BUS_NAMESPACE or AZURE_SERVICE_BUS_CONNECTION_STR."
        )
        correlation_id = f"REF-UNCONFIGURED-{uuid.uuid4().hex[:6].upper()}"
        answer = (
            "Escalation is not yet fully configured. "
            "Please contact your support team directly. "
            f"Reference: `{correlation_id}`."
        )

    return QueryResponse(
        question_id=f"q-{uuid.uuid4().hex[:12]}",
        answer_id=f"ans-{uuid.uuid4().hex[:12]}",
        conversation_id=conversation_id,
        user_id=user_id,
        status="ticket_raised",
        answer=answer,
        domain=domain,
        confidence=1.0,
        attempts_used=0,
        tools_used=[],
        sources=[],
        escalation_options=None,
    )


@step
async def handle_connect_sme(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
) -> QueryResponse:
    if is_escalation_configured():
        try:
            correlation_id = await asyncio.to_thread(
                sb_connect_sme,
                user_id, conversation_id, question_id, question_text, domain,
            )
            answer = (
                f"You're being connected with an SME. Reference: `{correlation_id}`. "
                f"Expected response within **{settings.ESCALATION_SLA_SME}**."
            )
            record_escalation(escalation_type="connect_sme", domain=domain or "unknown")
            logger.info(
                "sme_connect_queued correlation_id=%s user_id=%s domain=%s",
                correlation_id, user_id, domain,
            )
        except Exception as exc:
            logger.error("sme_queue_failed user_id=%s: %s", user_id, exc, exc_info=True)
            correlation_id = f"REF-PENDING-{uuid.uuid4().hex[:6].upper()}"
            answer = (
                "Your SME request was received but could not be queued automatically. "
                "Please contact the support team directly. "
                f"Reference: `{correlation_id}`."
            )
    else:
        logger.error(
            "sme_queue_skipped: Service Bus not configured. "
            "Set AZURE_SERVICE_BUS_NAMESPACE or AZURE_SERVICE_BUS_CONNECTION_STR."
        )
        correlation_id = f"REF-UNCONFIGURED-{uuid.uuid4().hex[:6].upper()}"
        answer = (
            "SME connection is not yet fully configured. "
            "Please contact your support team directly. "
            f"Reference: `{correlation_id}`."
        )

    return QueryResponse(
        question_id=f"q-{uuid.uuid4().hex[:12]}",
        answer_id=f"ans-{uuid.uuid4().hex[:12]}",
        conversation_id=conversation_id,
        user_id=user_id,
        status="sme_connecting",
        answer=answer,
        domain=domain,
        confidence=1.0,
        attempts_used=0,
        tools_used=[],
        sources=[],
        escalation_options=None,
    )


@workflow(name="main_agent_workflow")
async def main_agent_workflow(user_query: UserQuery) -> QueryResponse:
    text_lower = user_query.text.strip().lower()

    if text_lower == "raise_ticket":
        return await handle_raise_ticket(
            user_id=user_query.user_id,
            conversation_id=user_query.conversation_id,
            question_id=user_query.question_id,
            question_text=user_query.text,
            domain="",
        )
    if text_lower == "connect_sme":
        return await handle_connect_sme(
            user_id=user_query.user_id,
            conversation_id=user_query.conversation_id,
            question_id=user_query.question_id,
            question_text=user_query.text,
            domain="",
        )

    session = await load_session(user_query.conversation_id, user_query.user_id)
    ltm     = await load_ltm(user_query.user_id)

    try:
        final: FinalResponse = await call_orchestrator(OrchestratorInput(
            user_query=user_query,
            session_context=format_session_context(session, user_query.text),
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

    # Only true "error" (no usable model output at all) blanks the answer —
    # "failure" (low confidence after retries) still has a genuine LLM-written
    # answer worth showing, just without citations.
    has_answer = final.status != "error"
    show_escalation_options = final.status in ("failure", "error")
    domain_str = (
        final.domain.value.upper()
        if isinstance(final.domain, Domain)
        else (final.domain or "")
    )
    domain_metric = (final.domain.value if isinstance(final.domain, Domain) else (final.domain or "unknown")).lower()
    record_query(domain=domain_metric, status=final.status, tool=final.tools_used[-1] if final.tools_used else "")
    record_confidence(confidence=final.confidence, domain=domain_metric, status=final.status)
    record_attempts(attempts=final.attempts_used, domain=domain_metric, status=final.status)

    response = QueryResponse(
        question_id=user_query.question_id,
        answer_id=final.answer_id,
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        status=final.status,
        answer=final.answer if has_answer else "",
        domain=domain_str,
        confidence=final.confidence,
        attempts_used=final.attempts_used,
        tools_used=final.tools_used,
        sources=final.sources,
        escalation_options=_ESCALATION_OPTIONS if show_escalation_options else None,
        show_citations=final.show_citations,
        citations=final.citations,
    )

    # Persist to Cosmos synchronously before returning so feedback submitted
    # immediately after the response always finds a valid answer record.
    await asyncio.to_thread(upsert_document, get_chat_container(), ChatHistoryRecord(
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

    await append_turn(session, ConversationTurn(
        question_id=user_query.question_id,
        answer_id=final.answer_id,
        question=user_query.text,
        answer=final.answer,
        domain=domain_str,
        confidence=final.confidence,
        tools_used=final.tools_used,
    ))

    if len(session.turns) % settings.LTM_SUMMARY_EVERY_N == 0:
        task = asyncio.create_task(
            _run_ltm_update(user_query.user_id, session),
            name=f"ltm-update-{user_query.user_id}",
        )
        task.add_done_callback(_ltm_task_done_callback)

    return response


def _ltm_task_done_callback(task: asyncio.Task) -> None:
    """Catch unhandled exceptions from the LTM background task."""
    exc = task.exception() if not task.cancelled() else None
    if exc:
        logger.error(
            "ltm_task_unhandled_exception task=%s: %s",
            task.get_name(), exc, exc_info=exc,
        )


async def _run_ltm_update(user_id: str, session) -> None:
    """Run LTM update with a single retry on failure."""
    for attempt in range(1, 3):
        try:
            await update_ltm(user_id, session)
            return
        except Exception as exc:
            logger.error(
                "ltm_update_failed user_id=%s attempt=%d/%d: %s",
                user_id, attempt, 2, exc, exc_info=True,
            )
            if attempt < 2:
                await asyncio.sleep(5)


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _http
    _register_sigterm()

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_asyncio_exception_handler)

    _http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )
    await asyncio.to_thread(probe_cosmos)
    logger.info("main_agent_started environment=%s", settings.ENVIRONMENT)
    yield
    await _http.aclose()
    logger.info("main_agent_stopped")


def _register_sigterm():
    def _handler(signum, frame):
        logger.info("main_agent_sigterm_received — draining in-flight requests")
    signal.signal(signal.SIGTERM, _handler)


def _asyncio_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    exc = context.get("exception")
    logger.error(
        "asyncio_unhandled_exception msg=%s exc=%s",
        context.get("message"), exc, exc_info=exc,
    )


app = FastAPI(title="RAG Main Agent", lifespan=lifespan)

app.add_middleware(_ContentSizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ask-ops-bot-frontend-e9dfe0aqgfdcg7e3.southcentralus-01.azurewebsites.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health/live")
async def liveness() -> dict:
    """Always 200 while the process is alive — used for ACA liveness probe."""
    return {"status": "alive", "agent": "main"}


@app.get("/health/ready")
async def readiness() -> Response:
    """Returns 200 only when all dependencies are reachable — ACA readiness probe."""
    checks: dict[str, str] = {}
    overall_ok = True

    try:
        await asyncio.to_thread(get_chat_container().read)
        checks["cosmos"] = "ok"
    except Exception as exc:
        checks["cosmos"] = f"error: {type(exc).__name__}"
        overall_ok = False

    try:
        from shared.azure_clients import get_openai_client
        await asyncio.to_thread(get_openai_client().models.list)
        checks["openai"] = "ok"
    except Exception as exc:
        checks["openai"] = f"error: {type(exc).__name__}"
        overall_ok = False

    try:
        global _http
        probe_client = _http or httpx.AsyncClient(timeout=5.0)
        r = await probe_client.get(f"{_ORCHESTRATOR_URL}/health/live")
        if r.status_code == 200:
            checks["orchestrator"] = "ok"
        else:
            checks["orchestrator"] = f"status={r.status_code}"
            overall_ok = False
    except Exception as exc:
        checks["orchestrator"] = f"error: {type(exc).__name__}"
        overall_ok = False

    cb = _orchestrator_breaker.to_dict()
    checks["orchestrator_circuit"] = cb["state"]
    if cb["state"] == "open":
        overall_ok = False

    http_status = status.HTTP_200_OK if overall_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return Response(
        content=json.dumps({
            "status": "ready" if overall_ok else "degraded",
            "agent":  "main",
            "checks": checks,
        }),
        media_type="application/json",
        status_code=http_status,
    )


# Keep /health as an alias for readiness so existing probes keep working.
@app.get("/health")
async def health() -> Response:
    return await readiness()


# ── POST /query ────────────────────────────────────────────────────────────────

@app.post("/query")
async def query(body: QueryBody) -> Response:
    try:
        check_rate_limit(body.user_id)
    except RateLimitExceeded as exc:
        logger.warning(
            "rate_limit_exceeded user_id=%s retry_after=%.1f",
            body.user_id, exc.retry_after,
        )
        return Response(
            content=json.dumps({
                "error":       "rate_limit_exceeded",
                "retry_after": exc.retry_after,
                "message":     f"Too many requests. Please wait {exc.retry_after}s.",
            }),
            media_type="application/json",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(exc.retry_after)},
        )

    # Warn on potential prompt injection — do not reject (LLM guardrails handle it).
    if _INJECTION_PATTERNS.search(body.text):
        logger.warning(
            "potential_prompt_injection_detected user_id=%s text_preview=%.80s",
            body.user_id, body.text,
        )

    conversation_id = body.conversation_id or str(uuid.uuid4())

    # Idempotency — check Cosmos first so restarts don't lose the cache.
    if body.idempotency_key:
        cached = await asyncio.to_thread(
            get_document,
            get_chat_container(),
            body.idempotency_key,
            conversation_id,
        )
        if cached and cached.get("status") in ("success", "out_of_scope", "ticket_raised", "sme_connecting"):
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

    # Validate that the answer exists AND belongs to the submitting user.
    # This prevents a user from submitting feedback on someone else's answer
    # by modifying the card payload.
    existing_answer = await asyncio.to_thread(
        get_document, get_chat_container(), body.question_id, body.conversation_id
    )
    if not existing_answer:
        logger.warning(
            "feedback_invalid_answer_id question_id=%s user_id=%s",
            body.question_id, body.user_id,
        )
        return Response(
            content=json.dumps({"status": "error", "detail": "Answer not found."}),
            media_type="application/json",
            status_code=404,
        )
    if existing_answer.get("user_id") != body.user_id:
        logger.warning(
            "feedback_ownership_mismatch question_id=%s claimant=%s owner=%s",
            body.question_id, body.user_id, existing_answer.get("user_id"),
        )
        return Response(
            content=json.dumps({"status": "error", "detail": "Forbidden."}),
            media_type="application/json",
            status_code=403,
        )

    record = FeedbackRecord(
        id=f"fb-{body.answer_id}",   # fixed ID — Cosmos upsert overwrites on double-submit
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


# ── GET /feedback ──────────────────────────────────────────────────────────────

@app.get("/feedback")
async def feedback_get(
    answer_id:       str | None = Query(default=None),
    question_id:     str | None = Query(default=None),
    conversation_id: str | None = Query(default=None),
    user_id:         str        = Query(default="anonymous"),
    limit:           int        = Query(default=20, ge=1, le=100),
) -> Response:
    bind_context(agent="main", user_id=user_id)

    # feedback container is partitioned by /question_id.
    # Scope to a single partition whenever question_id is available.
    if question_id:
        cosmos_query = (
            "SELECT * FROM c WHERE c.question_id = @question_id AND c.type = 'feedback' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@question_id", "value": question_id},
            {"name": "@limit",       "value": limit},
        ]
        docs = query_documents(
            get_feedback_container(), cosmos_query, params,
            partition_key=question_id,
        )
    elif answer_id:
        # answer_id is not the partition key — cross-partition required.
        # This is an admin/analytics path; the WARNING in query_documents is intentional.
        cosmos_query = (
            "SELECT * FROM c WHERE c.answer_id = @answer_id AND c.type = 'feedback' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@answer_id", "value": answer_id},
            {"name": "@limit",     "value": limit},
        ]
        docs = query_documents(get_feedback_container(), cosmos_query, params)
    elif conversation_id:
        cosmos_query = (
            "SELECT * FROM c WHERE c.conversation_id = @conv_id AND c.type = 'feedback' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@conv_id", "value": conversation_id},
            {"name": "@limit",   "value": limit},
        ]
        docs = query_documents(get_feedback_container(), cosmos_query, params)
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
    logger.info(
        "chat_history_requested conversation_id=%s user_id=%s",
        conversation_id, user_id,
    )

    # chat-history is partitioned by /conversation_id.
    # Scope to a single partition when conversation_id is provided.
    if conversation_id:
        cosmos_query = (
            "SELECT * FROM c WHERE c.conversation_id = @conv_id AND c.type = 'chat_history' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@conv_id", "value": conversation_id},
            {"name": "@limit",   "value": limit},
        ]
        docs = query_documents(
            get_chat_container(), cosmos_query, params,
            partition_key=conversation_id,
        )
    else:
        # user_id query on a conversation_id-partitioned container → cross-partition.
        # This is an admin/analytics path.
        cosmos_query = (
            "SELECT * FROM c WHERE c.user_id = @user_id AND c.type = 'chat_history' "
            "ORDER BY c.timestamp DESC OFFSET 0 LIMIT @limit"
        )
        params = [
            {"name": "@user_id", "value": user_id},
            {"name": "@limit",   "value": limit},
        ]
        docs = query_documents(get_chat_container(), cosmos_query, params)

    clean = [{k: v for k, v in d.items() if not k.startswith("_")} for d in docs]
    return Response(
        content=json.dumps({"count": len(clean), "history": clean}),
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run("agents.main_agent:app", host="0.0.0.0", port=8000, reload=False)
