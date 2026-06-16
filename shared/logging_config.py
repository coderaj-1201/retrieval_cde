"""
Structured JSON logging + optional Azure Application Insights via OpenTelemetry.
Every log record carries: time, level, logger, agent, conversation_id, user_id, msg.
Use get_logger() everywhere. Use bind_context() at request entry points.
"""
from __future__ import annotations

import logging
import sys
import json
from contextvars import ContextVar
from typing import Any

from shared.config import settings

# ── Request-scoped context vars ───────────────────────────────────────────────
_ctx_agent: ContextVar[str]           = ContextVar("agent",           default="")
_ctx_conversation: ContextVar[str]    = ContextVar("conversation_id", default="")
_ctx_user: ContextVar[str]            = ContextVar("user_id",         default="")
_ctx_question: ContextVar[str]        = ContextVar("question_id",     default="")


def bind_context(
    agent: str = "",
    conversation_id: str = "",
    user_id: str = "",
    question_id: str = "",
) -> None:
    """Call at the top of each request handler to stamp all logs in that request."""
    if agent:           _ctx_agent.set(agent)
    if conversation_id: _ctx_conversation.set(conversation_id)
    if user_id:         _ctx_user.set(user_id)
    if question_id:     _ctx_question.set(question_id)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.agent           = _ctx_agent.get()
        record.conversation_id = _ctx_conversation.get()
        record.user_id         = _ctx_user.get()
        record.question_id     = _ctx_question.get()
        return True


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "time":            self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":           record.levelname,
            "logger":          record.name,
            "agent":           getattr(record, "agent", ""),
            "conversation_id": getattr(record, "conversation_id", ""),
            "user_id":         getattr(record, "user_id", ""),
            "question_id":     getattr(record, "question_id", ""),
            "msg":             record.getMessage(),
        }
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)
        return json.dumps(doc)


def configure_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    handler.addFilter(_ContextFilter())

    root = logging.getLogger()
    root.setLevel(level)
    # Remove default handlers to avoid duplicate plain-text output
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy Azure SDK loggers
    for noisy in ("azure.core.pipeline", "azure.identity", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if settings.APPLICATIONINSIGHTS_CONNECTION_STRING:
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor
            configure_azure_monitor(
                connection_string=settings.APPLICATIONINSIGHTS_CONNECTION_STRING,
                # Disable default instrumentations that add noise in a bot context;
                # keep only the ones that are useful for distributed tracing.
                disable_offline_storage=False,
            )
            logging.getLogger(__name__).info(
                "azure_monitor_configured connection_string_prefix=%s",
                settings.APPLICATIONINSIGHTS_CONNECTION_STRING[:20],
            )
            # Initialise OTel metric instruments NOW — after the global meter
            # provider has been replaced by configure_azure_monitor().
            from shared.telemetry import setup_meters
            setup_meters()
        except ImportError:
            logging.getLogger(__name__).warning(
                "azure-monitor-opentelemetry not installed — skipping App Insights. "
                "Add azure-monitor-opentelemetry to requirements.txt."
            )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
