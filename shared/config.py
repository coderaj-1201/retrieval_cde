"""
Application settings — validated via pydantic-settings at startup.

Auth notes:
  - AZURE_OPENAI_API_KEY  : set only in local dev. Production uses DefaultAzureCredential.
  - AZURE_SEARCH_API_KEY  : set only in local dev. Production uses DefaultAzureCredential.
  - COSMOS_KEY            : set only in local dev. Production uses DefaultAzureCredential.
  Removing any of these from the environment forces managed-identity auth — the
  correct production path.  Leaving them set in production will shadow managed
  identity silently, which is why they must be absent from ACA env vars on prod.

  INTERNAL_API_SECRET is required in staging/production (inter-agent auth header).
  Leave blank only in local development.
"""
from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING     = "staging"
    PRODUCTION  = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Deployment environment ─────────────────────────────────────────────────
    ENVIRONMENT: Environment = Environment.PRODUCTION

    # ── Azure AI Foundry ───────────────────────────────────────────────────────
    AZURE_FOUNDRY_PROJECT_ENDPOINT: AnyHttpUrl
    # Used for managed-identity auth (production).
    AZURE_OPENAI_ENDPOINT: AnyHttpUrl
    # Used for API-key auth (local dev). Foundry resources expose a separate
    # cognitiveservices.azure.com endpoint for key-based access.
    # Format: https://<hub-name>.cognitiveservices.azure.com/
    # Leave blank in production (managed identity uses AZURE_OPENAI_ENDPOINT).
    AZURE_OPENAI_COGNITIVESERVICES_ENDPOINT: AnyHttpUrl | None = None
    AZURE_OPENAI_CHAT_DEPLOYMENT: str      = "gpt-41-mini"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-3-large"
    AZURE_OPENAI_API_VERSION: str          = "2025-01-01-preview"
    # None → DefaultAzureCredential (managed identity). Must NOT be set in prod.
    AZURE_OPENAI_API_KEY: SecretStr | None = None

    # ── Azure AI Search ────────────────────────────────────────────────────────
    AZURE_SEARCH_ENDPOINT: AnyHttpUrl
    # None → DefaultAzureCredential (managed identity). Must NOT be set in prod.
    AZURE_SEARCH_API_KEY: SecretStr | None = None
    AZURE_SEARCH_INDEX: str               = "idx-rag"
    AZURE_SEARCH_SEMANTIC_CONFIG: str     = "rag-semantic-config"

    # ── Cosmos DB ──────────────────────────────────────────────────────────────
    COSMOS_ENDPOINT: AnyHttpUrl
    # None → DefaultAzureCredential (managed identity). Must NOT be set in prod.
    COSMOS_KEY: SecretStr | None           = None
    COSMOS_DATABASE: str                   = "csmsdb-aishrdsvcs-eus-prod"
    COSMOS_CONTAINER_CHAT: str             = "chat-history"
    COSMOS_CONTAINER_FEEDBACK: str         = "feedback"
    COSMOS_CONTAINER_SESSIONS: str         = "sessions"
    COSMOS_CONTAINER_LTM: str             = "long-term-memory"

    # ── Inter-agent auth ───────────────────────────────────────────────────────
    # Shared secret sent as X-Internal-Secret header between agents.
    # Required in staging/production. Can be left empty in local dev only.
    INTERNAL_API_SECRET: SecretStr | None  = None

    # ── Service Bus (escalation fallback) ─────────────────────────────────────
    # Production: set AZURE_SERVICE_BUS_NAMESPACE only (managed identity).
    # Local dev: set AZURE_SERVICE_BUS_CONNECTION_STR (connection string).
    # Service Bus is used ONLY when Zendesk is not configured.
    AZURE_SERVICE_BUS_NAMESPACE: str | None       = None
    AZURE_SERVICE_BUS_CONNECTION_STR: SecretStr | None = None
    SB_QUEUE_ESCALATION: str                      = "escalation-requests"

    # ── Zendesk (primary escalation channel) ──────────────────────────────────
    # Set all three to enable Zendesk ticket creation.
    # ZENDESK_SUBDOMAIN : the part before .zendesk.com (e.g. "mycompany")
    # ZENDESK_API_TOKEN : generated in Zendesk Admin → Apps → API → Token access
    # ZENDESK_USER_EMAIL: the agent account used for API calls (must have ticket:write)
    # ZENDESK_GROUP_ID_TICKET: optional group ID for ticket routing
    # ZENDESK_GROUP_ID_SME   : optional group ID for SME-connection routing
    ZENDESK_SUBDOMAIN:       str | None       = None
    ZENDESK_API_TOKEN:       SecretStr | None = None
    ZENDESK_USER_EMAIL:      str | None       = None
    ZENDESK_GROUP_ID_TICKET: int | None       = None
    ZENDESK_GROUP_ID_SME:    int | None       = None

    # ── RAG tuning ─────────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float          = Field(default=0.65, ge=0.0, le=1.0)
    # Per-citation bar — lower than the overall answer gate so that a source
    # contributing ~50% of a cross-document answer isn't dropped from citations.
    CITATION_CONFIDENCE_THRESHOLD: float = Field(default=0.40, ge=0.0, le=1.0)
    MAX_RETRIEVAL_ATTEMPTS: int  = Field(default=3,    ge=1,   le=5)
    RETRIEVAL_TOP_K: int         = Field(default=5,    ge=1,   le=20)
    SYNTHESIS_TEMPERATURE: float = Field(default=0.0,  ge=0.0, le=1.0)
    MAX_QUERY_LENGTH: int        = Field(default=2000, ge=50,  le=8000)
    # Total character budget for the context block sent to the synthesis LLM.
    # Prevents exceeding the model's context window on large document sets.
    # gpt-4o context is ~128k tokens; 12000 chars ≈ ~3000 tokens, leaving
    # plenty of room for the system prompt, question, and output.
    SYNTHESIS_MAX_CONTEXT_CHARS: int = Field(default=12000, ge=2000, le=40000)
    # Max source citations returned to the caller (additional sources still
    # contribute to the synthesis context, only the citation list is capped).
    SYNTHESIS_MAX_SOURCES: int   = Field(default=5,    ge=1,   le=10)
    # Hard cap on the final answer text length (chars) sent to the user.
    # ~3000 chars ≈ ~500 words — readable in Teams without excessive scrolling.
    SYNTHESIS_MAX_ANSWER_CHARS: int = Field(default=3000, ge=500, le=8000)
    # Max tokens the synthesis LLM may generate. Must be large enough to hold
    # the full JSON envelope (answer + citations + keys) without truncation.
    # Truncated JSON causes a JSONDecodeError → fallback to raw response blob.
    # 3000 tokens ≈ 12000 chars, comfortably covers SYNTHESIS_MAX_ANSWER_CHARS.
    SYNTHESIS_MAX_TOKENS: int = Field(default=3000, ge=500, le=8000)

    # ── Memory ─────────────────────────────────────────────────────────────────
    SESSION_MAX_TURNS: int       = Field(default=10,     ge=1,   le=50)
    SESSION_TTL_SECONDS: int     = Field(default=604800, ge=3600)
    LTM_SUMMARY_EVERY_N: int     = Field(default=5,      ge=1,   le=20)
    LTM_MAX_SUMMARY_CHARS: int   = Field(default=3000,   ge=500, le=10000)
    LTM_MAX_FACTS: int           = Field(default=10,     ge=3,   le=30)

    # ── Rate limiting ──────────────────────────────────────────────────────────
    # Set REDIS_URL to enable distributed (multi-replica) rate limiting.
    # Omit REDIS_URL to use the in-process token bucket (single-worker only).
    REDIS_URL: str | None = None
    RATE_LIMIT_RPM:   int = Field(default=20, ge=1,  le=600)
    RATE_LIMIT_BURST: int = Field(default=5,  ge=1,  le=50)

    # ── Domain classification ──────────────────────────────────────────────────
    DOMAIN_CONFIDENCE_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0)

    # ── Escalation SLAs ────────────────────────────────────────────────────────
    # Override these in .env when SLA commitments change — no code edits needed.
    ESCALATION_SLA_TICKET: str = "4 business hours"
    ESCALATION_SLA_SME:    str = "2 business hours"

    # ── Observability ──────────────────────────────────────────────────────────
    APPLICATIONINSIGHTS_CONNECTION_STRING: str | None = None
    LOG_LEVEL: str = "INFO"

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        import logging
        level = getattr(logging, v.upper(), None)
        if level is None:
            raise ValueError(f"Invalid LOG_LEVEL '{v}'. Must be DEBUG/INFO/WARNING/ERROR/CRITICAL.")
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
