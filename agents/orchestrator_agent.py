"""
Orchestrator Agent
==================
Classifies query (domain + confidence + tool), runs retry loop with tool
escalation. When domain confidence < DOMAIN_CONFIDENCE_THRESHOLD, fans out
retrieval to both primary and secondary domains in parallel, merges by score.

Phase-2 hardening:
  - LLM classification wrapped with @llm_retry
  - Shared httpx.AsyncClient reused across requests (not created per call)
  - /health/live + /health/ready split for ACA probes
  - H-9: classification failure returns error response (no silent OPS fallback)
  - InternalAuthMiddleware validates incoming requests from Main Agent
  - X-Internal-Secret header added to all outbound Retrieval calls
  - CircuitBreaker on Retrieval agent calls
  - SIGTERM handler for graceful shutdown
"""
from __future__ import annotations

import asyncio
import json
import signal
import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
try:
    from agent_framework import step, workflow
except Exception:
    from retrieval_pipeline.agent_framework import step, workflow
from fastapi import FastAPI, Request, Response

from shared.auth_middleware import InternalAuthMiddleware
from shared.azure_clients import get_openai_client
from shared.circuit_breaker import CircuitBreaker, CircuitOpenError
from shared.config import settings
from shared.cosmos_client import probe_cosmos
from shared.logging_config import bind_context, configure_logging, get_logger
from shared.models import (
    ClassifyInput, Domain, FinalResponse, OrchestratorInput,
    OrchestratorRequest, RetrievalResult, RetrievalTool, UserQuery,
)
from shared.retry import llm_retry
import os
from dotenv import load_dotenv
load_dotenv()

configure_logging()
logger = get_logger(__name__)

_TOOL_LADDER   = [RetrievalTool.HYBRID, RetrievalTool.HYDE, RetrievalTool.DECOMPOSITION]
_RETRIEVAL_URL = os.getenv("RETRIEVAL_URL")
_ALL_DOMAINS   = list(Domain)

# Shared HTTP client — initialised in lifespan, reused across all requests.
# Avoids creating a new TCP connection per LLM/agent call.
_http: httpx.AsyncClient | None = None

# Circuit breaker for the Retrieval agent.
_retrieval_breaker = CircuitBreaker(name="retrieval-agent", fail_max=3, reset_timeout=30)

_CLASSIFY_SYSTEM = """
Classify this enterprise query.

Return ONLY JSON:
{
  "domain": "hr|legal|it|ops",
  "domain_confidence": <0.0-1.0>,
  "secondary_domain": "hr|legal|it|ops|none",
  "tool": "hybrid|hyde|decomposition",
  "reason": "brief"
}

domain:
hr=people/leave/payroll/benefits
legal=contracts/compliance/GDPR/NDA
it=tech/infra/software/access
ops=operations/playbooks/race procedures/athlete guides/event rules/cutoff times/SOPs

domain_confidence:
0.9+=certain
<0.6=ambiguous

secondary_domain:
best alternate domain if confidence is low

tool:
hybrid=direct factual questions
hyde=vague/conceptual questions
decomposition=complex multi-part questions
"""


def _internal_headers() -> dict[str, str]:
    """Return auth headers for outbound internal calls."""
    secret = (
        settings.INTERNAL_API_SECRET.get_secret_value()
        if settings.INTERNAL_API_SECRET is not None
        else None
    )
    return {"X-Internal-Secret": secret} if secret else {}


class ClassifyResult:
    __slots__ = ("domain", "domain_confidence", "secondary_domain", "tool", "failed")

    def __init__(
        self,
        domain: Domain | None,
        domain_confidence: float,
        secondary_domain: Domain | None,
        tool: RetrievalTool,
        failed: bool = False,
    ) -> None:
        self.domain            = domain
        self.domain_confidence = domain_confidence
        self.secondary_domain  = secondary_domain
        self.tool              = tool
        self.failed            = failed


@step
async def classify_query(inp: ClassifyInput) -> ClassifyResult:
    memory_block = "\n\n".join(filter(None, [inp.ltm_context, inp.session_context]))
    user_content = (
        f"{memory_block}\n\nQuestion: {inp.query}"
        if memory_block else
        f"Question: {inp.query}"
    )

    @llm_retry
    def _call_llm():
        return get_openai_client().chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )

    try:
        resp = await asyncio.to_thread(_call_llm)
        raw  = json.loads(resp.choices[0].message.content)
        logger.info("raw_classification=%s", raw)
    except json.JSONDecodeError as exc:
        logger.error("classify_json_parse_error query=%.60s exc=%s", inp.query, exc)
        return ClassifyResult(None, 0.0, None, RetrievalTool.HYBRID, failed=True)
    except Exception as exc:
        logger.error("classify_llm_error query=%.60s: %s", inp.query, exc, exc_info=True)
        return ClassifyResult(None, 0.0, None, RetrievalTool.HYBRID, failed=True)

    domain_raw = (raw.get("domain") or "").lower()
    try:
        domain = Domain(domain_raw)
    except ValueError:
        logger.warning("unknown_domain value='%s'", domain_raw)
        return ClassifyResult(None, 0.0, None, RetrievalTool.HYBRID, failed=True)

    try:
        domain_confidence = float(raw.get("domain_confidence", 1.0))
        domain_confidence = max(0.0, min(1.0, domain_confidence))
    except (TypeError, ValueError):
        domain_confidence = 1.0

    secondary_domain: Domain | None = None
    sec_raw = (raw.get("secondary_domain") or "none").lower()
    if sec_raw not in ("none", ""):
        try:
            secondary_domain = Domain(sec_raw)
            if secondary_domain == domain:
                secondary_domain = None
        except ValueError:
            secondary_domain = None

    tool_raw = (raw.get("tool") or "hybrid").lower()
    try:
        tool = RetrievalTool(tool_raw)
    except ValueError:
        logger.warning("unknown_tool value='%s' defaulting=hybrid", tool_raw)
        tool = RetrievalTool.HYBRID

    logger.info(
        "classify_complete domain=%s confidence=%.2f secondary=%s tool=%s reason='%s'",
        domain, domain_confidence, secondary_domain or "none", tool, raw.get("reason", ""),
    )
    return ClassifyResult(domain, domain_confidence, secondary_domain, tool)


async def _call_retrieval(req: OrchestratorRequest) -> RetrievalResult:
    global _http
    payload = {
        "query":           req.query,
        "domain":          req.domain.value,
        "tool":            req.tool.value,
        "attempt":         req.attempt,
        "conversation_id": req.conversation_id,
        "user_id":         req.user_id,
        "question_id":     req.question_id,
    }
    client = _http or httpx.AsyncClient(timeout=60.0)
    resp = await client.post(
        f"{_RETRIEVAL_URL}/retrieve",
        json=payload,
        headers=_internal_headers(),
    )
    resp.raise_for_status()
    data = resp.json()
    domain_val = data.get("domain", "")
    try:
        domain = Domain(domain_val) if domain_val else req.domain
    except ValueError:
        domain = req.domain
    tool_val = data.get("tool", "")
    try:
        tool = RetrievalTool(tool_val) if tool_val else req.tool
    except ValueError:
        tool = req.tool
    return RetrievalResult(
        query=data.get("query", req.query),
        domain=domain,
        tool=tool,
        attempt=data.get("attempt", req.attempt),
        answer=data.get("answer", ""),
        confidence=float(data.get("confidence", 0.0)),
        sources=data.get("sources", []),
        conversation_id=data.get("conversation_id", req.conversation_id),
        user_id=data.get("user_id", req.user_id),
        question_id=data.get("question_id", req.question_id),
    )


@step
async def call_retrieval(req: OrchestratorRequest) -> RetrievalResult:
    try:
        return await _retrieval_breaker.call(_call_retrieval, req)
    except CircuitOpenError as exc:
        logger.error(
            "retrieval_circuit_open attempt=%d domain=%s retry_after=%.1f",
            req.attempt, req.domain, exc.retry_after,
        )
        raise
    except httpx.TimeoutException:
        logger.error("retrieval_timeout attempt=%d domain=%s tool=%s", req.attempt, req.domain, req.tool)
        raise
    except httpx.HTTPStatusError as exc:
        logger.error("retrieval_http_error status=%d attempt=%d", exc.response.status_code, req.attempt)
        raise
    except Exception as exc:
        logger.error("retrieval_unexpected_error attempt=%d: %s", req.attempt, exc, exc_info=True)
        raise


async def _call_retrieval_safe(req: OrchestratorRequest) -> RetrievalResult | None:
    """call_retrieval with exception swallowed — used in parallel fan-out."""
    try:
        return await call_retrieval(req)
    except Exception as exc:
        logger.error("retrieval_fanout_failed domain=%s: %s", req.domain, exc)
        return None


def _merge_retrieval_results(
    primary: RetrievalResult,
    secondary: RetrievalResult | None,
) -> RetrievalResult:
    if secondary is None:
        return primary

    base  = primary if primary.confidence >= secondary.confidence else secondary
    other = secondary if base is primary else primary

    seen_titles: set[str] = set()
    merged_sources: list[dict] = []
    for src in sorted(
        base.sources + other.sources,
        key=lambda s: s.get("relevance", 0.0),
        reverse=True,
    ):
        t = src.get("title", "")
        if t not in seen_titles:
            seen_titles.add(t)
            merged_sources.append(src)

    logger.info(
        "fanout_merge primary_conf=%.3f secondary_conf=%.3f merged_sources=%d",
        primary.confidence, secondary.confidence, len(merged_sources),
    )

    import dataclasses
    return dataclasses.replace(
        base,
        sources=merged_sources[:5],
        confidence=max(primary.confidence, secondary.confidence),
    )


@workflow(name="orchestrator_workflow")
async def orchestrator_workflow(inp: OrchestratorInput) -> FinalResponse:
    user_query      = inp.user_query
    session_context = inp.session_context
    ltm_context     = inp.ltm_context

    bind_context(
        agent="orchestrator",
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        question_id=user_query.question_id,
    )
    logger.info("orchestrator_started query_preview=%.80s", user_query.text)

    classification = await classify_query(ClassifyInput(
        query=user_query.text,
        session_context=session_context,
        ltm_context=ltm_context,
    ))

    # H-9: if classification failed entirely, return an error response immediately
    # rather than silently defaulting to OPS and retrieving the wrong documents.
    if classification.failed or classification.domain is None:
        logger.error(
            "classify_failed_returning_error query_preview=%.60s",
            user_query.text,
        )
        return FinalResponse(
            status="error",
            answer="Unable to classify your query. Please rephrase and try again.",
            domain=None,
            sources=[],
            confidence=0.0,
            attempts_used=0,
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            question_id=user_query.question_id,
            tools_used=[],
        )

    domain            = classification.domain
    secondary_domain  = classification.secondary_domain
    domain_confidence = classification.domain_confidence
    is_cross_domain   = (
        domain_confidence < settings.DOMAIN_CONFIDENCE_THRESHOLD
        and secondary_domain is not None
    )

    if is_cross_domain:
        logger.info(
            "cross_domain_fanout primary=%s secondary=%s confidence=%.2f",
            domain, secondary_domain, domain_confidence,
        )

    last_result: RetrievalResult | None = None
    tools_tried: list[str] = []

    for attempt_idx in range(settings.MAX_RETRIEVAL_ATTEMPTS):
        idx     = min(attempt_idx, len(_TOOL_LADDER) - 1)
        tool    = _TOOL_LADDER[idx]
        attempt = attempt_idx + 1
        tools_tried.append(tool.value)

        logger.info(
            "retrieval_attempt attempt=%d/%d domain=%s tool=%s cross_domain=%s",
            attempt, settings.MAX_RETRIEVAL_ATTEMPTS, domain, tool, is_cross_domain,
        )

        primary_req = OrchestratorRequest(
            query=user_query.text, domain=domain, tool=tool,
            attempt=attempt, conversation_id=user_query.conversation_id,
            user_id=user_query.user_id, question_id=user_query.question_id,
        )

        if is_cross_domain and secondary_domain:
            secondary_req = OrchestratorRequest(
                query=user_query.text, domain=secondary_domain, tool=tool,
                attempt=attempt, conversation_id=user_query.conversation_id,
                user_id=user_query.user_id, question_id=user_query.question_id,
            )
            primary_result, secondary_result = await asyncio.gather(
                _call_retrieval_safe(primary_req),
                _call_retrieval_safe(secondary_req),
            )
            if primary_result is None and secondary_result is None:
                logger.error("retrieval_fanout_both_failed attempt=%d", attempt)
                continue
            result = (
                secondary_result if primary_result is None
                else primary_result if secondary_result is None
                else _merge_retrieval_results(primary_result, secondary_result)
            )
        else:
            try:
                result = await call_retrieval(primary_req)
            except CircuitOpenError:
                # Circuit is open — no point retrying; return fast failure
                return FinalResponse(
                    status="error",
                    answer="Retrieval service is temporarily unavailable. Please try again shortly.",
                    domain=domain,
                    sources=[],
                    confidence=0.0,
                    attempts_used=attempt,
                    conversation_id=user_query.conversation_id,
                    user_id=user_query.user_id,
                    question_id=user_query.question_id,
                    tools_used=tools_tried,
                )
            except Exception as exc:
                logger.error("retrieval_failed attempt=%d: %s", attempt, exc)
                continue

        last_result = result
        logger.info(
            "retrieval_result attempt=%d confidence=%.3f passed=%s",
            attempt, result.confidence, result.passed,
        )

        if result.passed:
            logger.info(
                "orchestrator_success attempt=%d confidence=%.3f",
                attempt, result.confidence,
            )
            return FinalResponse(
                status="success",
                answer=result.answer,
                domain=domain,
                sources=result.sources,
                confidence=result.confidence,
                attempts_used=attempt,
                conversation_id=user_query.conversation_id,
                user_id=user_query.user_id,
                question_id=user_query.question_id,
                tools_used=tools_tried,
            )

        logger.warning(
            "confidence_below_threshold attempt=%d confidence=%.3f threshold=%.2f",
            attempt, result.confidence, settings.CONFIDENCE_THRESHOLD,
        )

    logger.error(
        "orchestrator_failed all_attempts=%d exhausted",
        settings.MAX_RETRIEVAL_ATTEMPTS,
    )
    return FinalResponse(
        status="failure",
        answer="",
        domain=domain,
        sources=last_result.sources if last_result else [],
        confidence=last_result.confidence if last_result else 0.0,
        attempts_used=settings.MAX_RETRIEVAL_ATTEMPTS,
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        question_id=user_query.question_id,
        tools_used=tools_tried,
    )


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _http
    _register_sigterm()
    _http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
    )
    await asyncio.to_thread(probe_cosmos)
    logger.info("orchestrator_agent_started environment=%s", settings.ENVIRONMENT)
    yield
    await _http.aclose()
    logger.info("orchestrator_agent_stopped")


def _register_sigterm():
    def _handler(signum, frame):
        logger.info("orchestrator_agent_sigterm_received — draining in-flight requests")
    signal.signal(signal.SIGTERM, _handler)


app = FastAPI(title="RAG Orchestrator Agent", lifespan=lifespan)
app.add_middleware(InternalAuthMiddleware)


@app.get("/health/live")
async def liveness() -> dict:
    return {"status": "alive", "agent": "orchestrator"}


@app.get("/health/ready")
async def readiness() -> Response:
    checks: dict[str, str] = {}
    overall_ok = True

    try:
        from shared.cosmos_client import get_chat_container
        await asyncio.to_thread(get_chat_container().read)
        checks["cosmos"] = "ok"
    except Exception as exc:
        checks["cosmos"] = f"error: {type(exc).__name__}"
        overall_ok = False

    # Include circuit breaker state in readiness so ACA can remove the
    # replica from rotation when the retrieval agent is consistently failing.
    cb_state = _retrieval_breaker.to_dict()
    checks["retrieval_circuit"] = cb_state["state"]
    if cb_state["state"] == "open":
        overall_ok = False

    return Response(
        content=json.dumps({
            "status": "ready" if overall_ok else "degraded",
            "agent":  "orchestrator",
            "checks": checks,
        }),
        media_type="application/json",
        status_code=200 if overall_ok else 503,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "agent": "orchestrator"}


@app.post("/orchestrate")
async def orchestrate(raw: Request) -> Response:
    body        = await raw.json()
    session_ctx = body.pop("session_context", "")
    ltm_ctx     = body.pop("ltm_context", "")

    user_query = UserQuery(
        text=body.get("text", ""),
        conversation_id=body.get("conversation_id", ""),
        user_id=body.get("user_id", ""),
        question_id=body.get("question_id", ""),
    )
    bind_context(
        agent="orchestrator",
        conversation_id=user_query.conversation_id,
        user_id=user_query.user_id,
        question_id=user_query.question_id,
    )

    try:
        result_obj = await orchestrator_workflow.run(OrchestratorInput(
            user_query=user_query,
            session_context=session_ctx,
            ltm_context=ltm_ctx,
        ))
        outputs = result_obj.get_outputs()
        final: FinalResponse = outputs[0] if outputs else FinalResponse(
            status="failure", answer="", domain=None,
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            question_id=user_query.question_id,
        )
    except Exception as exc:
        logger.error("orchestrate_endpoint_error: %s", exc, exc_info=True)
        final = FinalResponse(
            status="error", answer="", domain=None,
            conversation_id=user_query.conversation_id,
            user_id=user_query.user_id,
            question_id=user_query.question_id,
        )

    return Response(
        content=json.dumps(final.to_dict()),
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run(
        "agents.orchestrator_agent:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        timeout_graceful_shutdown=60,
    )
