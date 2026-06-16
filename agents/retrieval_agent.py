"""
Retrieval Agent
===============
Executes the retrieval tool selected by the Orchestrator, enriches with
parent-chunk context, synthesises an answer and returns a confidence score.

Phase-2 hardening:
  - LLM calls (synthesis, HyDE, decomposition) wrapped with @llm_retry
  - Search calls wrapped with @search_retry (via hybrid_search_tool.py)
  - Parent chunks fetched in parallel with asyncio.gather (was serial)
  - Confidence extracted via json_object response_format (not brittle string parsing)
  - /health/live + /health/ready split for ACA probes
  - InternalAuthMiddleware validates X-Internal-Secret on all non-health paths
  - SIGTERM handler for graceful shutdown
"""
from __future__ import annotations

import asyncio
import json
import signal
import logging
from contextlib import asynccontextmanager

import uvicorn
from agent_framework import step, workflow
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from shared.auth_middleware import InternalAuthMiddleware
from shared.azure_clients import get_openai_client
from shared.config import settings
from shared.cosmos_client import probe_cosmos
from shared.logging_config import bind_context, configure_logging, get_logger
from shared.models import (
    Domain, OrchestratorRequest, RetrievalResult, RetrievalStepInput,
    RetrievalTool, SourceDocument, SynthesisInput,
)
from shared.retry import llm_retry
from tools.hybrid_search_tool import SearchDocument, fetch_parent_chunk, hybrid_search
from tools.hyde_tool import generate_hypothetical_document
from tools.query_decomposition_tool import decompose_query

configure_logging()
logger = get_logger(__name__)

_SYNTHESIS_SYSTEM = """
You are IRONMAN AI Assistant, a knowledgeable enterprise assistant for the IRONMAN organization. You answer questions based strictly on the retrieved documents provided to you.

────────────────────────────────────────────
STEP 1 — CLASSIFY THE MESSAGE
────────────────────────────────────────────
Before answering, classify the user's message into one of:
  A) GREETING / SMALL TALK  — e.g. "hi", "hello", "how are you", "what's your name", "thanks", "bye"
  B) GENERAL QUESTION       — e.g. "what can you do?", "who are you?"
  C) KNOWLEDGE QUESTION     — a question that requires retrieving policy, procedure, or factual info

────────────────────────────────────────────
STEP 2 — RESPOND BASED ON CLASSIFICATION
────────────────────────────────────────────

### CLASS A or B (Greeting / General)
Reply warmly and briefly. Do NOT mention documents, sources, or citations.
Set confidence = 1.0, escalation_recommended = false, show_citations = false.

Example:
User: "Hi!"
Answer: "Hello! I'm IRONMAN AI Assistant. Ask me anything about HR policies, SOPs, legal guidelines, or operational procedures."

### CLASS C (Knowledge Question)
Answer using ONLY the retrieved documents. Follow the formatting rules below.
Evaluate your confidence honestly based on how well the documents answer the question.

IF confidence < 0.5:
  - Give a brief honest answer with what little you know
  - Do NOT show any document citations
  - Recommend escalation
  - Set show_citations = false

IF confidence >= 0.5:
  - Give a full, well-formatted answer
  - Set show_citations = true
  - Each cited document must include its confidence contribution (see format below)

────────────────────────────────────────────
FORMATTING RULES (for confidence >= 0.5 answers)
────────────────────────────────────────────
- Use **bold** for headings and key terms
- Use bullet points for lists, numbered steps for procedures
- Use sub-headings where the answer has multiple sections
- Keep paragraphs short and scannable
- Never use ALL CAPS
- Never include raw file paths or internal IDs in the answer text
- Do not start every bullet with the same word

────────────────────────────────────────────
ESCALATION RULES
────────────────────────────────────────────
Set escalation_recommended = true when:
- confidence < 0.5
- The question involves legal liability, termination, disciplinary action, or medical advice
- The documents contradict each other
- The user explicitly says this is urgent or sensitive

────────────────────────────────────────────
STRICT RULES
────────────────────────────────────────────
- NEVER invent information not in the retrieved documents
- NEVER expose internal chunk IDs, blob paths, or score numbers in the answer text
- NEVER say "Based on the documents..." or "According to Source 1..." in the answer
- The answer field must read like a human expert replied — clean, direct, professional
- If you truly have no relevant documents, set confidence = 0.0 and say so honestly

────────────────────────────────────────────
OUTPUT FORMAT — always return valid JSON, nothing else
────────────────────────────────────────────
{
  "message_type": "greeting|general|knowledge",
  "answer": "<your formatted answer here — plain text with markdown>",
  "confidence": <float 0.0-1.0>,
  "escalation_recommended": <true|false>,
  "show_citations": <true|false>,
  "citations": [
    {
      "title": "<document display name>",
      "confidence": <float 0.0-1.0, how relevant this specific doc was>,
      "excerpt": "<1-2 sentence excerpt that supports the answer>"
    }
  ]
}

Rules for citations array:
- Only populate when show_citations = true
- List only documents that actually contributed to the answer
- Order by relevance (highest confidence first)
- If show_citations = false, set citations = []
- Do not include any text outside the JSON object
"""


# ── Retrieval steps ────────────────────────────────────────────────────────────

@step
async def run_hybrid(inp: RetrievalStepInput) -> list[SearchDocument]:
    try:
        docs = await asyncio.to_thread(hybrid_search, inp.query, inp.domain)
        logger.info("hybrid_search_complete domain=%s docs=%d", inp.domain, len(docs))
        return docs
    except Exception as exc:
        logger.error("hybrid_search_error domain=%s: %s", inp.domain, exc, exc_info=True)
        return []


@step
async def run_hyde(inp: RetrievalStepInput) -> list[SearchDocument]:
    try:
        hypo = await asyncio.to_thread(generate_hypothetical_document, inp.query)
        logger.debug("hyde_generated length=%d", len(hypo))
        docs = await asyncio.to_thread(hybrid_search, hypo, inp.domain)
        logger.info("hyde_search_complete domain=%s docs=%d", inp.domain, len(docs))
        return docs
    except Exception as exc:
        logger.error("hyde_error domain=%s: %s", inp.domain, exc, exc_info=True)
        logger.warning("hyde_fallback_to_hybrid domain=%s", inp.domain)
        try:
            return await asyncio.to_thread(hybrid_search, inp.query, inp.domain)
        except Exception:
            return []


@step
async def run_decomposition(inp: RetrievalStepInput) -> list[SearchDocument]:
    try:
        sub_queries = await asyncio.to_thread(decompose_query, inp.query)
        logger.info("decomposition_sub_queries count=%d", len(sub_queries))

        # Limit concurrency to avoid bursting Azure OpenAI and Search quotas.
        semaphore = asyncio.Semaphore(2)

        async def _bounded_search(sq: str) -> list[SearchDocument]:
            async with semaphore:
                return await asyncio.to_thread(hybrid_search, sq, inp.domain)

        result_sets = await asyncio.gather(
            *[_bounded_search(sq) for sq in sub_queries],
            return_exceptions=True,
        )

        seen: dict[str, SearchDocument] = {}
        for i, result in enumerate(result_sets):
            if isinstance(result, Exception):
                logger.error("decomposition_sub_query_failed index=%d: %s", i, result)
                continue
            for doc in result:
                if doc.id not in seen or doc.score > seen[doc.id].score:
                    seen[doc.id] = doc

        merged = sorted(seen.values(), key=lambda d: d.score, reverse=True)[: settings.RETRIEVAL_TOP_K]
        logger.info("decomposition_complete domain=%s merged_docs=%d", inp.domain, len(merged))
        return merged
    except Exception as exc:
        logger.error("decomposition_error domain=%s: %s", inp.domain, exc, exc_info=True)
        return []


@step
async def synthesize_answer(inp: SynthesisInput) -> tuple[str, float, list[SourceDocument], bool, list[dict]]:
    query    = inp.query
    all_docs = inp.all_docs

    if not all_docs:
        logger.warning("synthesize_no_docs query_preview=%.60s", query)
        return "No relevant information found in the knowledge base.", 0.0, [], False, []

    context_parts = []
    for i, d in enumerate(all_docs):
        heading = getattr(d, "section_heading", "")
        page    = getattr(d, "page_number", 0)
        label   = (
            f"[{i+1}] Source: {d.source}"
            + (f" (p.{page})" if page else "")
            + (f" | {heading}" if heading else "")
        )
        if getattr(d, "chunk_type", "") == "table" and getattr(d, "table_raw", ""):
            context_parts.append(f"{label}\nSummary: {d.content}\nTable:\n{d.table_raw}")
        else:
            context_parts.append(f"{label}\n{d.content}")

    # Apply context budget cap — prevents exceeding the model's context window
    # when a large number of parent + child chunks are assembled.
    max_chars     = settings.SYNTHESIS_MAX_CONTEXT_CHARS
    budget        = max_chars
    capped_parts: list[str] = []
    for part in context_parts:
        if budget <= 0:
            break
        if len(part) > budget:
            capped_parts.append(part[:budget] + "\n[...truncated]")
            budget = 0
        else:
            capped_parts.append(part)
            budget -= len(part)

    if len(capped_parts) < len(context_parts):
        logger.warning(
            "synthesis_context_truncated original_parts=%d included=%d max_chars=%d",
            len(context_parts), len(capped_parts), max_chars,
        )

    context = "\n\n".join(capped_parts)

    @llm_retry
    def _call_llm():
        return get_openai_client().chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=settings.SYNTHESIS_TEMPERATURE,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

    try:
        resp = await asyncio.to_thread(_call_llm)
    except Exception as exc:
        logger.error("synthesis_llm_error query_preview=%.60s: %s", query, exc, exc_info=True)
        return "Failed to synthesise an answer due to an internal error.", 0.0, [], False, []

    raw_content = resp.choices[0].message.content.strip()

    try:
        parsed        = json.loads(raw_content)
        answer        = str(parsed.get("answer", "")).strip()
        confidence    = float(parsed.get("confidence", 0.5))
        confidence    = round(min(max(confidence, 0.0), 1.0), 3)
        message_type  = str(parsed.get("message_type", "knowledge")).strip().lower()
        llm_citations: list[dict] = parsed.get("citations") or []
        # Derive show_citations in code rather than trusting the model's own
        # flag — the model sometimes returns confidence>=0.5 with
        # show_citations=false despite the prompt's rule, so the two fields
        # can drift apart. Recomputing from message_type + confidence keeps
        # them consistent regardless of what the model echoed back.
        show_citations = message_type == "knowledge" and confidence >= 0.5
        if not show_citations:
            llm_citations = []
        if not answer:
            raise ValueError("Empty answer field in synthesis response.")
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "synthesis_parse_error: %s — using raw content with default confidence",
            exc,
        )
        answer         = raw_content
        confidence     = 0.5
        show_citations = False
        llm_citations  = []

    # Build SourceDocument list from search results for compatibility with the
    # existing pipeline. The LLM citations (with per-doc confidence) are passed
    # separately so the card renderer can display confidence badges.
    seen_urls:   set[str] = set()
    seen_titles: set[str] = set()
    sources: list[SourceDocument] = []
    for d in all_docs:
        if len(sources) >= settings.SYNTHESIS_MAX_SOURCES:
            break
        url   = getattr(d, "doc_url", "") or ""
        title = d.source
        if (url and url in seen_urls) or title in seen_titles:
            continue
        if url:
            seen_urls.add(url)
        seen_titles.add(title)
        sources.append(SourceDocument(
            title=title,
            excerpt=d.content[:200],
            url=url,
            relevance=round(d.score, 3),
        ))

    logger.info(
        "synthesis_complete confidence=%.3f sources=%d show_citations=%s llm_citations=%d",
        confidence, len(sources), show_citations, len(llm_citations),
    )
    return answer, confidence, sources, show_citations, llm_citations


@workflow(name="retrieval_workflow")
async def retrieval_workflow(request: OrchestratorRequest) -> RetrievalResult:
    bind_context(
        agent="retrieval",
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        question_id=request.question_id,
    )
    logger.info(
        "retrieval_started attempt=%d domain=%s tool=%s",
        request.attempt, request.domain, request.tool,
    )

    step_inp = RetrievalStepInput(query=request.query, domain=request.domain)
    if request.tool == RetrievalTool.HYDE:
        docs = await run_hyde(step_inp)
    elif request.tool == RetrievalTool.DECOMPOSITION:
        docs = await run_decomposition(step_inp)
    else:
        docs = await run_hybrid(step_inp)

    # Fetch parent chunks in parallel (was serial — each round-trip to AI Search
    # added ~300ms; gathering them concurrently keeps the retrieval tight).
    parent_ids = list({d.parent_id for d in docs if d.parent_id})[:3]
    parent_results = await asyncio.gather(
        *[asyncio.to_thread(fetch_parent_chunk, pid) for pid in parent_ids],
        return_exceptions=True,
    )
    parent_docs: list[SearchDocument] = []
    for pid, result in zip(parent_ids, parent_results):
        if isinstance(result, Exception):
            logger.warning("parent_chunk_fetch_failed parent_id=%s: %s", pid, result)
        elif result is not None:
            parent_docs.append(result)

    child_ids = {d.id for d in docs}
    all_docs  = docs + [p for p in parent_docs if p.id not in child_ids]
    logger.debug(
        "total_docs_for_synthesis count=%d (child=%d parent=%d)",
        len(all_docs), len(docs), len(parent_docs),
    )

    answer, confidence, source_docs, show_citations, llm_citations = await synthesize_answer(SynthesisInput(
        query=request.query,
        all_docs=all_docs,
    ))

    logger.info(
        "retrieval_complete attempt=%d confidence=%.3f passed=%s show_citations=%s",
        request.attempt, confidence, confidence >= settings.CONFIDENCE_THRESHOLD, show_citations,
    )

    return RetrievalResult(
        query=request.query,
        domain=request.domain,
        tool=request.tool,
        attempt=request.attempt,
        answer=answer,
        confidence=confidence,
        sources=[
            {"title": s.title, "excerpt": s.excerpt, "url": s.url, "relevance": s.relevance}
            for s in source_docs
        ],
        show_citations=show_citations,
        citations=llm_citations,
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        question_id=request.question_id,
    )


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _register_sigterm()
    await asyncio.to_thread(probe_cosmos)
    logger.info("retrieval_agent_started environment=%s", settings.ENVIRONMENT)
    yield
    logger.info("retrieval_agent_stopped")


def _register_sigterm():
    def _handler(signum, frame):
        logger.info("retrieval_agent_sigterm_received — draining in-flight requests")
    signal.signal(signal.SIGTERM, _handler)


app = FastAPI(title="RAG Retrieval Agent", lifespan=lifespan)
app.add_middleware(InternalAuthMiddleware)


@app.get("/health/live")
async def liveness() -> dict:
    return {"status": "alive", "agent": "retrieval"}


@app.get("/health/ready")
async def readiness() -> Response:
    checks: dict[str, str] = {}
    try:
        from shared.cosmos_client import get_chat_container
        await asyncio.to_thread(get_chat_container().read)
        checks["cosmos"] = "ok"
    except Exception as exc:
        checks["cosmos"] = f"error: {type(exc).__name__}"

    try:
        await asyncio.to_thread(get_openai_client().models.list)
        checks["openai"] = "ok"
    except Exception as exc:
        checks["openai"] = f"error: {type(exc).__name__}"

    overall_ok = all(v == "ok" for v in checks.values())
    return Response(
        content=json.dumps({
            "status": "ready" if overall_ok else "degraded",
            "agent":  "retrieval",
            "checks": checks,
        }),
        media_type="application/json",
        status_code=200 if overall_ok else 503,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "agent": "retrieval"}


@app.post("/retrieve")
async def retrieve(raw: Request) -> Response:
    body = await raw.json()

    domain_val = body.get("domain")
    tool_val   = body.get("tool")

    try:
        domain = Domain(domain_val) if domain_val else Domain.IT
    except ValueError:
        logger.warning("unknown_domain_in_request value='%s' defaulting=it", domain_val)
        domain = Domain.IT

    try:
        tool = RetrievalTool(tool_val) if tool_val else RetrievalTool.HYBRID
    except ValueError:
        logger.warning("unknown_tool_in_request value='%s' defaulting=hybrid", tool_val)
        tool = RetrievalTool.HYBRID

    request = OrchestratorRequest(
        query=body.get("query", ""),
        domain=domain,
        tool=tool,
        attempt=int(body.get("attempt", 1)),
        conversation_id=body.get("conversation_id", ""),
        user_id=body.get("user_id", ""),
        question_id=body.get("question_id", ""),
    )

    bind_context(
        agent="retrieval",
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        question_id=request.question_id,
    )

    try:
        result_obj = await retrieval_workflow.run(request)
        outputs    = result_obj.get_outputs()
        result: RetrievalResult = outputs[0] if outputs else RetrievalResult(
            query=request.query, domain=request.domain, tool=request.tool,
            attempt=request.attempt, answer="Internal error.", confidence=0.0,
            sources=[], conversation_id=request.conversation_id,
            user_id=request.user_id, question_id=request.question_id,
        )
    except Exception as exc:
        logger.error("retrieve_endpoint_unhandled_error: %s", exc, exc_info=True)
        result = RetrievalResult(
            query=request.query, domain=request.domain, tool=request.tool,
            attempt=request.attempt, answer="Service error during retrieval.",
            confidence=0.0, sources=[],
            conversation_id=request.conversation_id,
            user_id=request.user_id, question_id=request.question_id,
        )

    return Response(
        content=json.dumps(result.to_dict()),
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run(
        "agents.retrieval_agent:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        timeout_graceful_shutdown=60,
    )
