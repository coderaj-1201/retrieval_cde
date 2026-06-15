"""
Escalation client — sends escalation requests to Azure Service Bus.

Auth priority:
  1. AZURE_SERVICE_BUS_CONNECTION_STR set → connection string (local dev)
  2. AZURE_SERVICE_BUS_NAMESPACE set      → DefaultAzureCredential (production)

The client sends a JSON message to SB_QUEUE_ESCALATION and returns a
correlation_id immediately.  The Logic App subscribed to that queue handles
ticket creation (ServiceNow/Jira) and posts back via webhook when the real
ticket ID is available.

The `conversation_reference` field is stored on every escalation record so
that a future proactive-message flow can reach the user in Teams once the
real ticket ID arrives back.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from shared.config import settings

logger = logging.getLogger(__name__)


def _get_sender():
    """Return an azure-servicebus ServiceBusSender for the escalation queue."""
    from azure.servicebus import ServiceBusClient  # type: ignore[import-untyped]
    from azure.identity import DefaultAzureCredential

    conn_str: str | None = (
        settings.AZURE_SERVICE_BUS_CONNECTION_STR.get_secret_value()
        if settings.AZURE_SERVICE_BUS_CONNECTION_STR is not None
        else None
    )
    if conn_str:
        logger.debug("service_bus_auth=connection_string")
        sb_client = ServiceBusClient.from_connection_string(conn_str)
    elif settings.AZURE_SERVICE_BUS_NAMESPACE:
        logger.debug("service_bus_auth=managed_identity namespace=%s", settings.AZURE_SERVICE_BUS_NAMESPACE)
        sb_client = ServiceBusClient(
            fully_qualified_namespace=settings.AZURE_SERVICE_BUS_NAMESPACE,
            credential=DefaultAzureCredential(),
        )
    else:
        raise RuntimeError(
            "No Service Bus configuration found. "
            "Set AZURE_SERVICE_BUS_CONNECTION_STR (dev) or "
            "AZURE_SERVICE_BUS_NAMESPACE (prod)."
        )

    return sb_client.get_queue_sender(queue_name=settings.SB_QUEUE_ESCALATION)


def is_escalation_configured() -> bool:
    """Return True if Service Bus is configured — used to gate escalation paths."""
    return bool(
        settings.AZURE_SERVICE_BUS_CONNECTION_STR
        or settings.AZURE_SERVICE_BUS_NAMESPACE
    )


def raise_ticket(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
    conversation_reference: dict | None = None,
) -> str:
    """
    Send a ticket-creation request to Service Bus.
    Returns a correlation_id that the caller shows to the user as a
    provisional reference (e.g. "REF-abc123"). The real ticket ID
    will arrive via the Logic App webhook callback.

    Raises RuntimeError if Service Bus is not configured.
    """
    from azure.servicebus import ServiceBusMessage  # type: ignore[import-untyped]

    correlation_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
    payload = {
        "type":                   "raise_ticket",
        "correlation_id":         correlation_id,
        "user_id":                user_id,
        "conversation_id":        conversation_id,
        "question_id":            question_id,
        "question_text":          question_text[:1000],  # cap to avoid huge messages
        "domain":                 domain,
        "timestamp":              datetime.now(timezone.utc).isoformat(),
        "conversation_reference": conversation_reference or {},
    }

    sender = _get_sender()
    with sender:
        msg = ServiceBusMessage(
            body=json.dumps(payload),
            content_type="application/json",
            subject="raise_ticket",
            message_id=correlation_id,
            session_id=user_id,        # group by user for ordered processing
        )
        sender.send_messages(msg)

    logger.info(
        "escalation_ticket_queued correlation_id=%s user_id=%s domain=%s",
        correlation_id, user_id, domain,
    )
    return correlation_id


def connect_sme(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
    conversation_reference: dict | None = None,
) -> str:
    """
    Send an SME-connection request to Service Bus.
    Returns a correlation_id shown to the user as a provisional reference.
    """
    from azure.servicebus import ServiceBusMessage  # type: ignore[import-untyped]

    correlation_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
    payload = {
        "type":                   "connect_sme",
        "correlation_id":         correlation_id,
        "user_id":                user_id,
        "conversation_id":        conversation_id,
        "question_id":            question_id,
        "question_text":          question_text[:1000],
        "domain":                 domain,
        "timestamp":              datetime.now(timezone.utc).isoformat(),
        "conversation_reference": conversation_reference or {},
    }

    sender = _get_sender()
    with sender:
        msg = ServiceBusMessage(
            body=json.dumps(payload),
            content_type="application/json",
            subject="connect_sme",
            message_id=correlation_id,
            session_id=user_id,
        )
        sender.send_messages(msg)

    logger.info(
        "escalation_sme_queued correlation_id=%s user_id=%s domain=%s",
        correlation_id, user_id, domain,
    )
    return correlation_id
