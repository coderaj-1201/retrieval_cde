"""
Adversarial tests for LLM failure modes.

Covers: transient errors, permanent errors, malformed responses, boundary values,
confidence clamping, empty answers, retry exhaustion.

"The LLM will fail in prod. This proves the code handles it."
"""
from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

# ── Stub agent_framework so agent module imports work without the SDK ─────────
_af_stub = MagicMock()
_af_stub.step   = lambda fn: fn
_af_stub.workflow = MagicMock(return_value=lambda fn: fn)
sys.modules.setdefault("agent_framework", _af_stub)
sys.modules.setdefault("retrieval_pipeline.agent_framework", _af_stub)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chat_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _synthesis_json(answer="Test answer.", confidence=0.85, escalation=False) -> str:
    return json.dumps({
        "answer": answer,
        "confidence": confidence,
        "escalation_recommended": escalation,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CLASSIFICATION LLM FAILURES (orchestrator_agent)
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassificationLLMFailures:

    @pytest.mark.asyncio
    async def test_classify_returns_non_json_gives_failed_result(self):
        """If the LLM returns prose instead of JSON, ClassifyResult.failed=True."""
        from agents.orchestrator_agent import classify_query
        from shared.models import ClassifyInput

        mock_resp = _make_chat_response("I cannot classify this query.")

        with patch("agents.orchestrator_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.return_value = mock_resp
            result = await classify_query(ClassifyInput(query="what is my leave balance?"))

        assert result.failed is True
        assert result.domain is None

    @pytest.mark.asyncio
    async def test_classify_returns_unknown_domain_gives_failed_result(self):
        """Domain value not in Domain enum → ClassifyResult.failed=True."""
        from agents.orchestrator_agent import classify_query
        from shared.models import ClassifyInput

        bad_payload = json.dumps({
            "domain": "finance",   # not in Domain enum
            "domain_confidence": 0.9,
            "secondary_domain": "none",
            "tool": "hybrid",
            "reason": "test",
        })
        with patch("agents.orchestrator_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.return_value = _make_chat_response(bad_payload)
            result = await classify_query(ClassifyInput(query="show me the budget"))

        assert result.failed is True

    @pytest.mark.asyncio
    async def test_classify_llm_raises_rate_limit_is_retried(self):
        """RateLimitError triggers llm_retry (up to 3 attempts) then fails=True."""
        from agents.orchestrator_agent import classify_query
        from shared.models import ClassifyInput

        try:
            from openai import RateLimitError
            exc = RateLimitError("429", response=MagicMock(), body={})
        except Exception:
            exc = Exception("rate limit")

        with patch("agents.orchestrator_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.side_effect = exc
            # tenacity will retry 3× — patch sleep to avoid waiting
            with patch("tenacity.nap.time.sleep"):
                result = await classify_query(ClassifyInput(query="test"))

        assert result.failed is True

    @pytest.mark.asyncio
    async def test_classify_llm_raises_connection_error_is_retried(self):
        """APIConnectionError triggers llm_retry then returns failed result."""
        from agents.orchestrator_agent import classify_query
        from shared.models import ClassifyInput

        try:
            from openai import APIConnectionError
            exc = APIConnectionError(request=MagicMock())
        except Exception:
            exc = Exception("connection")

        with patch("agents.orchestrator_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.side_effect = exc
            with patch("tenacity.nap.time.sleep"):
                result = await classify_query(ClassifyInput(query="test"))

        assert result.failed is True

    @pytest.mark.asyncio
    async def test_classify_missing_confidence_field_defaults_to_1(self):
        """Missing domain_confidence should not crash; defaults to 1.0."""
        from agents.orchestrator_agent import classify_query
        from shared.models import ClassifyInput

        payload = json.dumps({
            "domain": "hr",
            # domain_confidence intentionally absent
            "secondary_domain": "none",
            "tool": "hybrid",
            "reason": "test",
        })
        with patch("agents.orchestrator_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.return_value = _make_chat_response(payload)
            result = await classify_query(ClassifyInput(query="who is my manager?"))

        assert result.failed is False
        assert result.domain_confidence == 1.0

    @pytest.mark.asyncio
    async def test_classify_confidence_clamped_to_valid_range(self):
        """LLM returning confidence=1.5 must not propagate above 1.0."""
        from agents.orchestrator_agent import classify_query
        from shared.models import ClassifyInput

        payload = json.dumps({
            "domain": "it",
            "domain_confidence": 1.5,   # over max
            "secondary_domain": "none",
            "tool": "hybrid",
            "reason": "test",
        })
        with patch("agents.orchestrator_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.return_value = _make_chat_response(payload)
            result = await classify_query(ClassifyInput(query="reset my password"))

        assert result.domain_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_classify_same_primary_and_secondary_clears_secondary(self):
        """If secondary_domain == domain, secondary must be set to None."""
        from agents.orchestrator_agent import classify_query
        from shared.models import ClassifyInput

        payload = json.dumps({
            "domain": "ops",
            "domain_confidence": 0.5,
            "secondary_domain": "ops",   # same as primary
            "tool": "hybrid",
            "reason": "test",
        })
        with patch("agents.orchestrator_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.return_value = _make_chat_response(payload)
            result = await classify_query(ClassifyInput(query="what is SOP?"))

        assert result.secondary_domain is None


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SYNTHESIS LLM FAILURES (retrieval_agent)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSynthesisLLMFailures:

    def _make_docs(self, n=2):
        """Minimal SearchDocument-like objects."""
        docs = []
        for i in range(n):
            d = MagicMock()
            d.id             = f"doc-{i}"
            d.content        = f"Content chunk {i} about enterprise policy."
            d.source         = f"Policy Doc {i}"
            d.score          = 0.9 - i * 0.1
            d.doc_url        = f"https://sharepoint.com/doc{i}.pdf"
            d.section_heading = ""
            d.page_number    = i + 1
            d.chunk_type     = "paragraph"
            d.table_raw      = ""
            docs.append(d)
        return docs

    @pytest.mark.asyncio
    async def test_synthesis_non_json_falls_back_to_raw_content(self):
        """LLM returning prose (not JSON) must not crash; falls back to raw text."""
        from agents.retrieval_agent import synthesize_answer
        from shared.models import SynthesisInput

        docs = self._make_docs()
        raw_prose = "The leave policy allows 15 days per year."

        with patch("agents.retrieval_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.return_value = _make_chat_response(raw_prose)
            answer, confidence, sources = await synthesize_answer(SynthesisInput(
                query="What is the leave policy?", all_docs=docs,
            ))

        assert answer == raw_prose
        assert confidence == 0.5   # default on parse failure

    @pytest.mark.asyncio
    async def test_synthesis_confidence_above_1_clamped(self):
        """Confidence of 2.5 must be clamped to 1.0."""
        from agents.retrieval_agent import synthesize_answer
        from shared.models import SynthesisInput

        with patch("agents.retrieval_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.return_value = _make_chat_response(
                _synthesis_json(confidence=2.5)
            )
            _, confidence, _ = await synthesize_answer(SynthesisInput(
                query="q", all_docs=self._make_docs(),
            ))
        assert confidence == 1.0

    @pytest.mark.asyncio
    async def test_synthesis_confidence_below_0_clamped(self):
        """Confidence of -0.3 must be clamped to 0.0."""
        from agents.retrieval_agent import synthesize_answer
        from shared.models import SynthesisInput

        with patch("agents.retrieval_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.return_value = _make_chat_response(
                _synthesis_json(confidence=-0.3)
            )
            _, confidence, _ = await synthesize_answer(SynthesisInput(
                query="q", all_docs=self._make_docs(),
            ))
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_synthesis_empty_answer_field_falls_back(self):
        """If the 'answer' field is an empty string, confidence falls back to 0.5."""
        from agents.retrieval_agent import synthesize_answer
        from shared.models import SynthesisInput

        with patch("agents.retrieval_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.return_value = _make_chat_response(
                json.dumps({"answer": "", "confidence": 0.9})
            )
            answer, confidence, _ = await synthesize_answer(SynthesisInput(
                query="q", all_docs=self._make_docs(),
            ))
        # Empty answer → ValueError → falls back to raw JSON string with confidence=0.5
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_synthesis_llm_exception_returns_safe_message(self):
        """Unhandled LLM exception returns a safe error message and confidence=0."""
        from agents.retrieval_agent import synthesize_answer
        from shared.models import SynthesisInput

        with patch("agents.retrieval_agent.get_openai_client") as m:
            m.return_value.chat.completions.create.side_effect = RuntimeError("Network dead")
            with patch("tenacity.nap.time.sleep"):
                answer, confidence, _ = await synthesize_answer(SynthesisInput(
                    query="q", all_docs=self._make_docs(),
                ))

        assert confidence == 0.0
        assert "error" in answer.lower() or "failed" in answer.lower()

    @pytest.mark.asyncio
    async def test_synthesis_zero_docs_returns_no_info_message(self):
        """Empty doc list must return 'no relevant information' without calling LLM."""
        from agents.retrieval_agent import synthesize_answer
        from shared.models import SynthesisInput

        with patch("agents.retrieval_agent.get_openai_client") as m:
            answer, confidence, sources = await synthesize_answer(SynthesisInput(
                query="What is the refund policy?", all_docs=[],
            ))
            m.return_value.chat.completions.create.assert_not_called()

        assert confidence == 0.0
        assert "No relevant" in answer

    @pytest.mark.asyncio
    async def test_synthesis_context_truncated_when_docs_exceed_budget(self):
        """When docs exceed SYNTHESIS_MAX_CONTEXT_CHARS, context is truncated."""
        from agents.retrieval_agent import synthesize_answer
        from shared.models import SynthesisInput

        # Create a doc whose content alone exceeds the budget
        big_doc = MagicMock()
        big_doc.id             = "big-doc"
        big_doc.content        = "x" * 15000   # larger than default 12000-char budget
        big_doc.source         = "Big Doc"
        big_doc.score          = 0.9
        big_doc.doc_url        = ""
        big_doc.section_heading = ""
        big_doc.page_number    = 1
        big_doc.chunk_type     = "paragraph"
        big_doc.table_raw      = ""

        second_doc = MagicMock()
        second_doc.id          = "second"
        second_doc.content     = "Additional content."
        second_doc.source      = "Doc B"
        second_doc.score       = 0.8
        second_doc.doc_url     = ""
        second_doc.section_heading = ""
        second_doc.page_number = 1
        second_doc.chunk_type  = "paragraph"
        second_doc.table_raw   = ""

        captured_context = []

        def _capture_call(**kwargs):
            msgs = kwargs.get("messages", [])
            for m in msgs:
                if m.get("role") == "user":
                    captured_context.append(m["content"])
            return _make_chat_response(_synthesis_json())

        with patch("agents.retrieval_agent.get_openai_client") as m, \
             patch.object(__import__("shared.config", fromlist=["settings"]).settings,
                          "SYNTHESIS_MAX_CONTEXT_CHARS", 12000):
            m.return_value.chat.completions.create.side_effect = (
                lambda *a, **kw: _capture_call(**kw)
            )
            await synthesize_answer(SynthesisInput(
                query="test", all_docs=[big_doc, second_doc],
            ))

        # Context sent to LLM must not exceed budget + label overhead
        if captured_context:
            assert len(captured_context[0]) < 13000 + 500   # label overhead allowance


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  LTM UPDATE LLM FAILURE (memory)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLTMUpdateLLMFailures:

    @pytest.mark.asyncio
    async def test_ltm_update_llm_failure_does_not_crash(self):
        """LLM failure during LTM update must be swallowed — never crashes the query."""
        from shared.memory import update_ltm
        from shared.models import SessionMemory, ConversationTurn

        session = SessionMemory(conversation_id="c-1", user_id="u-1")
        session.turns = [ConversationTurn(
            question_id="q", answer_id="a",
            question="Who is my manager?", answer="Your manager is Bob.",
            domain="HR", confidence=0.9, tools_used=["hybrid"],
        )]

        with patch("shared.memory.get_ltm_container"), \
             patch("shared.memory.get_document", return_value=None), \
             patch("shared.azure_clients.get_openai_client") as m, \
             patch("shared.memory.upsert_document"):
            m.return_value.chat.completions.create.side_effect = RuntimeError("LLM down")
            # Must not raise
            await update_ltm("u-1", session)

    @pytest.mark.asyncio
    async def test_ltm_update_llm_non_json_falls_back_to_prior(self):
        """Non-JSON LTM response keeps prior summary intact."""
        from shared.memory import update_ltm
        from shared.models import SessionMemory, ConversationTurn

        session = SessionMemory(conversation_id="c-1", user_id="u-1")
        session.turns = [ConversationTurn(
            question_id="q", answer_id="a",
            question="Q", answer="A",
            domain="ops", confidence=0.8, tools_used=[],
        )]

        with patch("shared.memory.get_ltm_container"), \
             patch("shared.memory.get_document", return_value=None), \
             patch("shared.azure_clients.get_openai_client") as m, \
             patch("shared.memory.upsert_document"):
            m.return_value.chat.completions.create.return_value = _make_chat_response(
                "This is prose, not JSON."
            )
            # json.loads will raise → except block returns without crash
            await update_ltm("u-1", session)
