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
    AZURE_OPENAI_ENDPOINT: AnyHttpUrl
    AZURE_OPENAI_CHAT_DEPLOYMENT: str      = "gpt-4o"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-ada-002"
    AZURE_OPENAI_API_VERSION: str          = "2024-12-01-preview"
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

    # ── Service Bus (escalation) ───────────────────────────────────────────────
    # Production: set AZURE_SERVICE_BUS_NAMESPACE only (managed identity).
    # Local dev: set AZURE_SERVICE_BUS_CONNECTION_STR (connection string).
    AZURE_SERVICE_BUS_NAMESPACE: str | None       = None
    AZURE_SERVICE_BUS_CONNECTION_STR: SecretStr | None = None
    SB_QUEUE_ESCALATION: str                      = "escalation-requests"

    # ── RAG tuning ─────────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float  = Field(default=0.75, ge=0.0, le=1.0)
    MAX_RETRIEVAL_ATTEMPTS: int  = Field(default=3,    ge=1,   le=5)
    RETRIEVAL_TOP_K: int         = Field(default=5,    ge=1,   le=20)
    SYNTHESIS_TEMPERATURE: float = Field(default=0.0,  ge=0.0, le=1.0)
    MAX_QUERY_LENGTH: int        = Field(default=2000, ge=50,  le=8000)

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
