"""
Custom OpenTelemetry metrics for the RAG pipeline.

IMPORTANT — initialisation order:
  Meters must be created AFTER configure_azure_monitor() has replaced the global
  meter provider, otherwise they bind to the default NoOp provider and export
  nothing.  Call setup_meters() from logging_config.configure_logging() once
  Azure Monitor is wired up.  All recording functions are silent no-ops until
  setup_meters() has been called, so they are safe to call unconditionally
  throughout the codebase.

Metrics exported to Application Insights:
  rag.query.count          Counter    — queries by domain / status / tool
  rag.query.confidence     Histogram  — confidence score per successful query
  rag.retrieval.attempts   Histogram  — number of attempts before success/failure
  rag.escalation.count     Counter    — escalations by type (ticket/sme) and domain
  rag.tool.count           Counter    — retrieval tool usage by tool name and domain
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Instruments — None until setup_meters() is called.
_query_counter       = None
_confidence_histo    = None
_attempts_histo      = None
_escalation_counter  = None
_tool_counter        = None


def setup_meters() -> None:
    """
    Initialise OTel instruments against the current global meter provider.

    Must be called AFTER configure_azure_monitor() (or any other provider setup)
    so the instruments bind to the correct exporter.  Safe to call multiple times
    (subsequent calls are no-ops).
    """
    global _query_counter, _confidence_histo, _attempts_histo
    global _escalation_counter, _tool_counter

    if _query_counter is not None:
        return   # already initialised

    try:
        from opentelemetry import metrics as _m

        meter = _m.get_meter("rag-bot", version="1.0")

        _query_counter = meter.create_counter(
            name="rag.query.count",
            description="Total queries processed, labelled by domain, status, and tool.",
            unit="1",
        )
        _confidence_histo = meter.create_histogram(
            name="rag.query.confidence",
            description="Confidence score distribution for completed queries.",
            unit="1",
        )
        _attempts_histo = meter.create_histogram(
            name="rag.retrieval.attempts",
            description="Number of retrieval attempts used before final result.",
            unit="1",
        )
        _escalation_counter = meter.create_counter(
            name="rag.escalation.count",
            description="Escalation events by type and domain.",
            unit="1",
        )
        _tool_counter = meter.create_counter(
            name="rag.tool.count",
            description="Retrieval tool invocations by tool name and domain.",
            unit="1",
        )
        logger.info("telemetry_meters_initialised provider=%s", type(meter).__name__)

    except Exception as exc:
        logger.warning("telemetry_meters_setup_failed: %s", exc)


# ── Public recording helpers ───────────────────────────────────────────────────
# All functions are silent no-ops when setup_meters() has not been called yet
# or if the OTel SDK is unavailable.  Callers need no guards.

def record_query(*, domain: str, status: str, tool: str = "") -> None:
    """Increment the query counter. Call once per completed /query request."""
    if _query_counter is None:
        return
    try:
        _query_counter.add(1, {"domain": domain, "status": status, "tool": tool})
    except Exception as exc:
        logger.debug("telemetry_record_query_error: %s", exc)


def record_confidence(*, confidence: float, domain: str, status: str) -> None:
    """Record confidence score histogram. Call after synthesis completes."""
    if _confidence_histo is None:
        return
    try:
        _confidence_histo.record(
            max(0.0, min(1.0, confidence)),
            {"domain": domain, "status": status},
        )
    except Exception as exc:
        logger.debug("telemetry_record_confidence_error: %s", exc)


def record_attempts(*, attempts: int, domain: str, status: str) -> None:
    """Record how many retrieval attempts were needed. Call after orchestrator loop."""
    if _attempts_histo is None:
        return
    try:
        _attempts_histo.record(attempts, {"domain": domain, "status": status})
    except Exception as exc:
        logger.debug("telemetry_record_attempts_error: %s", exc)


def record_escalation(*, escalation_type: str, domain: str) -> None:
    """Increment escalation counter. Call when a ticket or SME connection is raised."""
    if _escalation_counter is None:
        return
    try:
        _escalation_counter.add(1, {"type": escalation_type, "domain": domain})
    except Exception as exc:
        logger.debug("telemetry_record_escalation_error: %s", exc)


def record_tool(*, tool: str, domain: str) -> None:
    """Increment tool-usage counter. Call each time a retrieval tool is dispatched."""
    if _tool_counter is None:
        return
    try:
        _tool_counter.add(1, {"tool": tool, "domain": domain})
    except Exception as exc:
        logger.debug("telemetry_record_tool_error: %s", exc)
