"""
Local dev settings — validated via pydantic-settings.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Azure AI Foundry ──────────────────────────────────────────────────────
    AZURE_FOUNDRY_PROJECT_ENDPOINT: AnyHttpUrl
    AZURE_OPENAI_ENDPOINT: AnyHttpUrl
    AZURE_OPENAI_CHAT_DEPLOYMENT: str      = "gpt-4o"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-ada-002"
    AZURE_OPENAI_API_VERSION: str          = "2024-12-01-preview"
    AZURE_OPENAI_API_KEY: SecretStr | None = None   # optional — uses DefaultAzureCredential if not set

    # ── Azure AI Search ───────────────────────────────────────────────────────
    AZURE_SEARCH_ENDPOINT: AnyHttpUrl
    AZURE_OPENAI_ENDPOINT: AnyHttpUrl
    AZURE_SEARCH_API_KEY: SecretStr
    AZURE_SEARCH_INDEX: str               = "idx-rag"
    AZURE_SEARCH_SEMANTIC_CONFIG: str     = "rag-semantic-config"

    # ── Cosmos DB ─────────────────────────────────────────────────────────────
    COSMOS_ENDPOINT: AnyHttpUrl
    COSMOS_KEY: SecretStr
    COSMOS_DATABASE: str                  = "csmsdb-aishrdsvcs-eus-prod"
    COSMOS_CONTAINER_CHAT: str            = "chat-history"     # per-turn Q&A + full response
    COSMOS_CONTAINER_FEEDBACK: str        = "feedback"         # /feedback posts
    COSMOS_CONTAINER_SESSIONS: str        = "sessions"         # short-term memory (conversation turns)
    COSMOS_CONTAINER_LTM: str             = "long-term-memory" # per-user summarised memory

    # ── RAG tuning ────────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float  = Field(default=0.75, ge=0.0, le=1.0)
    MAX_RETRIEVAL_ATTEMPTS: int  = Field(default=3,    ge=1,   le=5)
    RETRIEVAL_TOP_K: int         = Field(default=5,    ge=1,   le=20)
    SYNTHESIS_TEMPERATURE: float = Field(default=0.0,  ge=0.0, le=1.0)

    # ── Memory ────────────────────────────────────────────────────────────────
    SESSION_MAX_TURNS: int       = Field(default=10,      ge=1,   le=50)    # short-term window size
    SESSION_TTL_SECONDS: int     = Field(default=604800,  ge=3600)          # 7 days; written to Cosmos doc ttl field
    LTM_SUMMARY_EVERY_N: int     = Field(default=5,       ge=1,   le=20)    # compress LTM every N turns

    # ── Rate limiting (token bucket, per user_id, single-process) ────────────
    RATE_LIMIT_RPM:   int = Field(default=20,  ge=1,  le=600)  # tokens refilled per minute
    RATE_LIMIT_BURST: int = Field(default=5,   ge=1,  le=50)   # max burst size

    # ── Domain classification ─────────────────────────────────────────────────
    # Below this confidence the orchestrator fans out to the secondary domain in parallel
    DOMAIN_CONFIDENCE_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0)

    # ── Observability ─────────────────────────────────────────────────────────
    APPLICATIONINSIGHTS_CONNECTION_STRING: str | None = None
    LOG_LEVEL: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
