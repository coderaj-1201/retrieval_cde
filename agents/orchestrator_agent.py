"""
Orchestrator Agent
==================
Classifies query (domain + confidence + tool), runs retry loop with tool
escalation. When domain confidence < DOMAIN_CONFIDENCE_THRESHOLD, fans out
retrieval to both primary and secondary domains in parallel, merges by score,
and synthesises once from the combined context.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
try:
    from agent_framework import step, workflow
except Exception:
    from retrieval_pipeline.agent_framework import step, workflow
from fastapi import FastAPI, Request, Response

from shared.azure_clients import get_openai_client
from shared.config import settings
from shared.cosmos_client import probe_cosmos
from shared.logging_config import bind_context, configure_logging, get_logger
from shared.models import (
    ClassifyInput, Domain, FinalResponse, OrchestratorInput,
    OrchestratorRequest, RetrievalResult, RetrievalTool, UserQuery,
)
import os
from dotenv import load_dotenv
load_dotenv()

configure_logging()
logger = get_logger(__name__)

_TOOL_LADDER    = [RetrievalTool.HYBRID, RetrievalTool.HYDE, RetrievalTool.DECOMPOSITION]
_RETRIEVAL_URL  = os.getenv("RETRIEVAL_URL")
_ALL_DOMAINS    = list(Domain)

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

# ── Dataclass for classification result ───────────────────────────────────────

class ClassifyResult:
    __slots__ = ("domain", "domain_confidence", "secondary_domain", "tool")

    def __init__(
        self,
        domain: Domain,
        domain_confidence: float,
        secondary_domain: Domain | None,
        tool: RetrievalTool,
    ) -> None:
        self.domain             = domain
        self.domain_confidence  = domain_confidence
        self.secondary_domain   = secondary_domain
        self.tool               = tool


@step
async def classify_query(inp: ClassifyInput) -> ClassifyResult:
    memory_block = "\n\n".join(filter(None, [inp.ltm_context, inp.session_context]))
    user_content = f"{memory_block}\n\nQuestion: {inp.query}" if memory_block else f"Question: {inp.query}"

    try:
        resp = await asyncio.to_thread(
            get_openai_client().chat.completions.create,
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content)

        logger.info("RAW_CLASSIFICATION=%s", raw)
    except json.JSONDecodeError as exc:
        logger.error("classify_json_parse_error query=%.60s exc=%s", inp.query, exc)
        return ClassifyResult(Domain.OPS, 1.0, None, RetrievalTool.HYBRID)
    except Exception as exc:
        logger.error("classify_llm_error query=%.60s exc=%s", inp.query, exc, exc_info=True)
        return ClassifyResult(Domain.OPS, 1.0, None, RetrievalTool.HYBRID)

    # Domain
    domain_raw = (raw.get("domain") or "ops").lower()
    try:
        domain = Domain(domain_raw)
    except ValueError:
        logger.warning("unknown_domain value='%s' defaulting=ops", domain_raw)
        domain = Domain.IT

    # Domain confidence
    try:
        domain_confidence = float(raw.get("domain_confidence", 1.0))
        domain_confidence = max(0.0, min(1.0, domain_confidence))
    except (TypeError, ValueError):
        domain_confidence = 1.0

    # Secondary domain (only meaningful when confidence is low)
    secondary_domain: Domain | None = None
    sec_raw = (raw.get("secondary_domain") or "none").lower()
    if sec_raw not in ("none", ""):
        try:
            secondary_domain = Domain(sec_raw)
            if secondary_domain == domain:
                secondary_domain = None
        except ValueError:
            secondary_domain = None

    # Tool
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


@step
async def call_retrieval(req: OrchestratorRequest) -> RetrievalResult:
    payload = {
        "query":           req.query,
        "domain":          req.domain.value,
        "tool":            req.tool.value,
        "attempt":         req.attempt,
        "conversation_id": req.conversation_id,
        "user_id":         req.user_id,
        "question_id":     req.question_id,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{_RETRIEVAL_URL}/retrieve", json=payload)
            resp.raise_for_status()
            return RetrievalResult(**resp.json())
    except httpx.TimeoutException:
        logger.error(
            "retrieval_timeout attempt=%d domain=%s tool=%s",
            req.attempt, req.domain, req.tool,
        )
        raise
    except httpx.HTTPStatusError as exc:
        logger.error(
            "retrieval_http_error status=%d attempt=%d: %s",
            exc.response.status_code, req.attempt, exc,
        )
        raise
    except Exception as exc:
        logger.error(
            "retrieval_unexpected_error attempt=%d: %s",
            req.attempt, exc, exc_info=True,
        )
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
    """
    Merge primary + secondary domain results for cross-domain queries.
    Interleaves sources by relevance score, deduplicates by title.
    Returns a new RetrievalResult with merged sources and the higher confidence.
    The actual synthesis happens in the retrieval agent — we just pick the
    better answer (highest confidence) and annotate it with all sources.
    """
    if secondary is None:
        return primary

    # Pick the answer with higher confidence as the base
    base   = primary if primary.confidence >= secondary.confidence else secondary
    other  = secondary if base is primary else primary

    # Merge sources — deduplicate by title, sort by relevance descending
    seen_titles: set[str] = set()
    merged_sources: list[dict] = []
    all_sources = sorted(
        base.sources + other.sources,
        key=lambda s: s.get("relevance", 0.0),
        reverse=True,
    )
    for src in all_sources:
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
    domain              = classification.domain
    secondary_domain    = classification.secondary_domain
    domain_confidence   = classification.domain_confidence
    is_cross_domain     = (
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
            # F4: parallel retrieval across both domains
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
            # If one side failed, fall back to the other
            if primary_result is None:
                result = secondary_result
            elif secondary_result is None:
                result = primary_result
            else:
                result = _merge_retrieval_results(primary_result, secondary_result)
        else:
            try:
                result = await call_retrieval(primary_req)
            except Exception as exc:
                logger.error("retrieval_failed attempt=%d: %s", attempt, exc)
                continue

        last_result = result
        logger.info(
            "retrieval_result attempt=%d confidence=%.3f passed=%s",
            attempt, result.confidence, result.passed,
        )

        if result.passed:
            logger.info("orchestrator_success attempt=%d confidence=%.3f", attempt, result.confidence)
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

    logger.error("orchestrator_failed all_attempts=%d exhausted", settings.MAX_RETRIEVAL_ATTEMPTS)
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
    # B1 fix: probe_cosmos is blocking — run in thread pool
    await asyncio.to_thread(probe_cosmos)
    logger.info("orchestrator_agent_started")
    yield
    logger.info("orchestrator_agent_stopped")


app = FastAPI(title="RAG Orchestrator Agent", lifespan=lifespan)


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
        outputs    = result_obj.get_outputs()
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
    uvicorn.run("agents.orchestrator_agent:app", host="0.0.0.0", port=8001, reload=False)
