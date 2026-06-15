"""
Custom OpenTelemetry metrics for the RAG pipeline.

Activated automatically when APPLICATIONINSIGHTS_CONNECTION_STRING is set
(configure_azure_monitor is called in logging_config.configure_logging()).
When the connection string is absent, all recording functions are silent no-ops
so this module is safe to call unconditionally everywhere.

Metrics exported:
  rag.query.count          Counter    — queries by domain / status / tool
  rag.query.confidence     Histogram  — confidence score per successful query
  rag.retrieval.attempts   Histogram  — number of attempts before success/failure
  rag.escalation.count     Counter    — escalations by type (ticket/sme) and domain
  rag.tool.count           Counter    — retrieval tool usage by tool name and domain
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics as _otel_metrics

    _meter = _otel_metrics.get_meter("rag-bot", version="1.0")

    _query_counter = _meter.create_counter(
        name="rag.query.count",
        description="Total queries processed, labelled by domain, status, and tool.",
        unit="1",
    )
    _confidence_histogram = _meter.create_histogram(
        name="rag.query.confidence",
        description="Confidence score distribution for completed queries.",
        unit="1",
    )
    _attempts_histogram = _meter.create_histogram(
        name="rag.retrieval.attempts",
        description="Number of retrieval attempts used before final result.",
        unit="1",
    )
    _escalation_counter = _meter.create_counter(
        name="rag.escalation.count",
        description="Escalation events by type and domain.",
        unit="1",
    )
    _tool_counter = _meter.create_counter(
        name="rag.tool.count",
        description="Retrieval tool invocations by tool name and domain.",
        unit="1",
    )
    _METRICS_AVAILABLE = True

except Exception as _import_exc:  # pragma: no cover
    logger.warning("telemetry_metrics_unavailable: %s", _import_exc)
    _METRICS_AVAILABLE = False


# ── Public recording helpers ───────────────────────────────────────────────────
# All functions are no-ops when metrics are unavailable — callers need no guards.

def record_query(
    *,
    domain: str,
    status: str,
    tool: str = "",
) -> None:
    """Increment the query counter. Call once per completed /query request."""
    if not _METRICS_AVAILABLE:
        return
    try:
        _query_counter.add(1, {"domain": domain, "status": status, "tool": tool})
    except Exception as exc:
        logger.debug("telemetry_record_query_error: %s", exc)


def record_confidence(
    *,
    confidence: float,
    domain: str,
    status: str,
) -> None:
    """Record confidence score histogram. Call after synthesis completes."""
    if not _METRICS_AVAILABLE:
        return
    try:
        _confidence_histogram.record(
            max(0.0, min(1.0, confidence)),
            {"domain": domain, "status": status},
        )
    except Exception as exc:
        logger.debug("telemetry_record_confidence_error: %s", exc)


def record_attempts(
    *,
    attempts: int,
    domain: str,
    status: str,
) -> None:
    """Record how many retrieval attempts were needed. Call after orchestrator loop."""
    if not _METRICS_AVAILABLE:
        return
    try:
        _attempts_histogram.record(attempts, {"domain": domain, "status": status})
    except Exception as exc:
        logger.debug("telemetry_record_attempts_error: %s", exc)


def record_escalation(
    *,
    escalation_type: str,
    domain: str,
) -> None:
    """Increment escalation counter. Call when a ticket or SME connection is queued."""
    if not _METRICS_AVAILABLE:
        return
    try:
        _escalation_counter.add(1, {"type": escalation_type, "domain": domain})
    except Exception as exc:
        logger.debug("telemetry_record_escalation_error: %s", exc)


def record_tool(
    *,
    tool: str,
    domain: str,
) -> None:
    """Increment tool-usage counter. Call each time a retrieval tool is dispatched."""
    if not _METRICS_AVAILABLE:
        return
    try:
        _tool_counter.add(1, {"tool": tool, "domain": domain})
    except Exception as exc:
        logger.debug("telemetry_record_tool_error: %s", exc)
