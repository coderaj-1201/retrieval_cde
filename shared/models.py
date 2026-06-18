"""
Shared typed models for the RAG pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4


# ── Enums ─────────────────────────────────────────────────────────────────────

class Domain(StrEnum):
    HR    = "hr"
    LEGAL = "legal"
    IT    = "it"
    OPS   = "ops"


# Human-readable descriptions for each domain.
# Used to build the LLM classification prompt dynamically so that adding a
# new domain only requires adding it here + to the enum.
DOMAIN_DESCRIPTIONS: dict[str, str] = {
    Domain.HR:    "people / leave / payroll / benefits / recruitment / performance",
    Domain.LEGAL: "contracts / compliance / GDPR / NDA / regulatory / IP",
    Domain.IT:    "tech / infrastructure / software / access / security / systems",
    Domain.OPS:   "operations / playbooks / procedures / event rules / SLAs / cutoff times / SOPs / athlete guides",
}


class RetrievalTool(StrEnum):
    HYBRID        = "hybrid"
    HYDE          = "hyde"
    DECOMPOSITION = "decomposition"


class FeedbackRating(StrEnum):
    THUMBS_UP   = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    NEUTRAL     = "neutral"


# ── MAF input wrappers (single-param constraint on @step / @workflow) ─────────

@dataclass
class UserQuery:
    text:            str
    conversation_id: str
    user_id:         str
    question_id:     str = field(default_factory=lambda: f"q-{uuid4().hex[:12]}")


@dataclass
class OrchestratorInput:
    """Single input for MAF @workflow orchestrator_workflow."""
    user_query:      UserQuery
    session_context: str = ""
    ltm_context:     str = ""


@dataclass
class ClassifyInput:
    """Single input for MAF @step classify_query."""
    query:           str
    session_context: str = ""
    ltm_context:     str = ""


@dataclass
class RetrievalStepInput:
    """Single input for MAF @step run_hybrid / run_hyde / run_decomposition."""
    query:  str
    domain: str


@dataclass
class SynthesisInput:
    """Single input for MAF @step synthesize_answer."""
    query:           str
    all_docs:        list   # list[SearchDocument] — avoid circular import with tools
    session_context: str = ""  # injected only for follow-up queries


# ── Core pipeline models ───────────────────────────────────────────────────────

@dataclass
class OrchestratorRequest:
    query:           str
    domain:          Domain
    tool:            RetrievalTool
    attempt:         int
    conversation_id: str
    user_id:         str
    question_id:     str = ""
    session_context: str = ""  # forwarded to synthesis for follow-up queries


@dataclass
class SourceDocument:
    title:     str
    excerpt:   str
    url:       str   = ""
    relevance: float = 0.0


@dataclass
class RetrievalResult:
    query:           str
    domain:          Domain
    tool:            RetrievalTool
    attempt:         int
    answer:          str
    confidence:      float
    sources:         list[dict]
    conversation_id: str
    user_id:         str
    question_id:     str        = ""
    show_citations:  bool       = False
    citations:       list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        from shared.config import settings
        return self.confidence >= settings.CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "query":           self.query,
            "domain":          self.domain.value if isinstance(self.domain, Domain) else (self.domain or ""),
            "tool":            self.tool.value if isinstance(self.tool, RetrievalTool) else (self.tool or ""),
            "attempt":         self.attempt,
            "answer":          self.answer,
            "confidence":      self.confidence,
            "sources":         self.sources,
            "show_citations":  self.show_citations,
            "citations":       self.citations,
            "conversation_id": self.conversation_id,
            "user_id":         self.user_id,
            "question_id":     self.question_id,
        }


@dataclass
class FinalResponse:
    status:        str            # "success" | "failure" | "error"
    answer:        str
    domain:        Domain | None
    sources:       list[dict]  = field(default_factory=list)
    confidence:    float       = 0.0
    attempts_used: int         = 0
    conversation_id: str       = ""
    user_id:       str         = ""
    question_id:   str         = ""
    answer_id:     str         = field(default_factory=lambda: f"ans-{uuid4().hex[:12]}")
    tools_used:    list[str]   = field(default_factory=list)
    show_citations: bool       = False
    citations:      list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status":          self.status,
            "answer":          self.answer,
            "domain":          self.domain.value if isinstance(self.domain, Domain) else (self.domain or ""),
            "sources":         self.sources,
            "confidence":      self.confidence,
            "attempts_used":   self.attempts_used,
            "conversation_id": self.conversation_id,
            "user_id":         self.user_id,
            "question_id":     self.question_id,
            "answer_id":       self.answer_id,
            "tools_used":      [t.value if isinstance(t, RetrievalTool) else t for t in self.tools_used],
            "show_citations":  self.show_citations,
            "citations":       self.citations,
        }


# ── API response models ────────────────────────────────────────────────────────

@dataclass
class QueryResponse:
    question_id:        str
    answer_id:          str
    conversation_id:    str
    user_id:            str
    status:             str
    answer:             str
    domain:             str
    confidence:         float
    attempts_used:      int
    tools_used:         list[str]
    sources:            list[dict]
    escalation_options: dict | None
    show_citations:     bool       = False
    citations:          list[dict] = field(default_factory=list)
    timestamp:          str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "question_id":        self.question_id,
            "answer_id":          self.answer_id,
            "conversation_id":    self.conversation_id,
            "user_id":            self.user_id,
            "status":             self.status,
            "answer":             self.answer,
            "domain":             self.domain,
            "confidence":         self.confidence,
            "attempts_used":      self.attempts_used,
            "tools_used":         self.tools_used,
            "sources":            self.sources,
            "escalation_options": self.escalation_options,
            "show_citations":     self.show_citations,
            "citations":          self.citations,
            "timestamp":          self.timestamp,
        }


# ── Feedback models ────────────────────────────────────────────────────────────

@dataclass
class FeedbackRecord:
    id:              str
    question_id:     str
    answer_id:       str
    user_id:         str
    conversation_id: str
    rating:          str
    comment:         str
    timestamp:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "question_id":     self.question_id,
            "answer_id":       self.answer_id,
            "user_id":         self.user_id,
            "conversation_id": self.conversation_id,
            "rating":          self.rating,
            "comment":         self.comment,
            "timestamp":       self.timestamp,
            "type":            "feedback",
        }


# ── Memory models ─────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    question_id: str
    answer_id:   str
    question:    str
    answer:      str
    domain:      str
    confidence:  float
    tools_used:  list[str]
    timestamp:   str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "answer_id":   self.answer_id,
            "question":    self.question,
            "answer":      self.answer,
            "domain":      self.domain,
            "confidence":  self.confidence,
            "tools_used":  self.tools_used,
            "timestamp":   self.timestamp,
        }


@dataclass
class SessionMemory:
    conversation_id: str
    user_id:         str
    turns:           list[ConversationTurn] = field(default_factory=list)
    created_at:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        from shared.config import settings
        return {
            "id":              self.conversation_id,
            "conversation_id": self.conversation_id,
            "user_id":         self.user_id,
            "turns":           [t.to_dict() for t in self.turns],
            "created_at":      self.created_at,
            "updated_at":      self.updated_at,
            "type":            "session",
            "ttl":             settings.SESSION_TTL_SECONDS,
        }


@dataclass
class LongTermMemoryRecord:
    id:                      str
    user_id:                 str
    summary:                 str
    key_facts:               list[str]
    last_updated:            str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_conversation_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id":                      self.id,
            "user_id":                 self.user_id,
            "summary":                 self.summary,
            "key_facts":               self.key_facts,
            "last_updated":            self.last_updated,
            "source_conversation_ids": self.source_conversation_ids,
            "type":                    "long_term_memory",
        }


# ── Chat history model ────────────────────────────────────────────────────────

@dataclass
class ChatHistoryRecord:
    id:              str
    conversation_id: str
    user_id:         str
    question_id:     str
    answer_id:       str
    question:        str
    answer:          str
    domain:          str
    confidence:      float
    tools_used:      list[str]
    sources:         list[dict]
    status:          str
    timestamp:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "conversation_id": self.conversation_id,
            "user_id":         self.user_id,
            "question_id":     self.question_id,
            "answer_id":       self.answer_id,
            "question":        self.question,
            "answer":          self.answer,
            "domain":          self.domain,
            "confidence":      self.confidence,
            "tools_used":      self.tools_used,
            "sources":         self.sources,
            "status":          self.status,
            "timestamp":       self.timestamp,
            "type":            "chat_history",
        }
