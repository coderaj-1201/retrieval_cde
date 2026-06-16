"""
Escalation client — creates support tickets via Zendesk (primary) or
Azure Service Bus (fallback when Zendesk is not configured).

Priority:
  1. Zendesk configured (ZENDESK_SUBDOMAIN + ZENDESK_API_TOKEN + ZENDESK_USER_EMAIL)
     → ticket created via Zendesk REST API; real Zendesk ticket ID returned.
  2. Service Bus configured (AZURE_SERVICE_BUS_NAMESPACE or CONNECTION_STR)
     → message enqueued for downstream Logic App / webhook processing.
  3. Neither configured → RuntimeError (surfaced as a 503 to the caller).

The caller always receives a reference string:
  - Zendesk path: "ZD-{ticket_id}"   (real Zendesk ticket number)
  - Service Bus path: "REF-{hex}"     (provisional correlation ID)

Zendesk field mapping:
  raise_ticket → ticket with group_id=ZENDESK_GROUP_ID_TICKET, tag "raise_ticket"
  connect_sme  → ticket with group_id=ZENDESK_GROUP_ID_SME,    tag "connect_sme"
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import httpx

from shared.config import settings

logger = logging.getLogger(__name__)


# ── Zendesk ───────────────────────────────────────────────────────────────────

def _zendesk_configured() -> bool:
    return bool(
        settings.ZENDESK_SUBDOMAIN
        and settings.ZENDESK_API_TOKEN
        and settings.ZENDESK_USER_EMAIL
    )


def _zendesk_create_ticket(
    *,
    subject: str,
    body: str,
    tags: list[str],
    external_id: str,
    group_id: int | None,
    requester_email: str | None = None,
) -> str:
    """
    POST a ticket to Zendesk and return its ID as "ZD-{id}".
    Raises RuntimeError on HTTP errors (caller logs and surfaces to user).
    """
    ticket: dict = {
        "subject":     subject,
        "comment":     {"body": body},
        "tags":        tags,
        "external_id": external_id,
        # Priority normal by default; Zendesk automation can escalate based on tags.
        "priority":    "normal",
    }
    if requester_email:
        ticket["requester"] = {"email": requester_email}
    if group_id:
        ticket["group_id"] = group_id

    subdomain = settings.ZENDESK_SUBDOMAIN
    user_email = settings.ZENDESK_USER_EMAIL
    api_token = settings.ZENDESK_API_TOKEN.get_secret_value()  # type: ignore[union-attr]

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"https://{subdomain}.zendesk.com/api/v2/tickets",
                auth=(f"{user_email}/token", api_token),
                json={"ticket": ticket},
                headers={"Content-Type": "application/json"},
            )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Zendesk API error {exc.response.status_code}: "
            f"{exc.response.text[:300]}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Zendesk request failed: {exc}") from exc

    ticket_id = resp.json()["ticket"]["id"]
    return f"ZD-{ticket_id}"


# ── Service Bus (fallback) ─────────────────────────────────────────────────────

def _sb_configured() -> bool:
    return bool(
        settings.AZURE_SERVICE_BUS_CONNECTION_STR
        or settings.AZURE_SERVICE_BUS_NAMESPACE
    )


def _sb_get_sender():
    from azure.servicebus import ServiceBusClient  # type: ignore[import-untyped]
    from azure.identity import DefaultAzureCredential

    conn_str: str | None = (
        settings.AZURE_SERVICE_BUS_CONNECTION_STR.get_secret_value()
        if settings.AZURE_SERVICE_BUS_CONNECTION_STR is not None
        else None
    )
    if conn_str:
        sb_client = ServiceBusClient.from_connection_string(conn_str)
    else:
        sb_client = ServiceBusClient(
            fully_qualified_namespace=settings.AZURE_SERVICE_BUS_NAMESPACE,
            credential=DefaultAzureCredential(),
        )
    return sb_client.get_queue_sender(queue_name=settings.SB_QUEUE_ESCALATION)


def _sb_send(escalation_type: str, payload: dict, correlation_id: str) -> None:
    from azure.servicebus import ServiceBusMessage  # type: ignore[import-untyped]

    sender = _sb_get_sender()
    with sender:
        msg = ServiceBusMessage(
            body=json.dumps(payload),
            content_type="application/json",
            subject=escalation_type,
            message_id=correlation_id,
            session_id=payload.get("user_id", ""),
        )
        try:
            sender.send_messages(msg)
        except Exception as send_exc:
            logger.error(
                "escalation_sb_send_failed type=%s correlation_id=%s "
                "user_id=%s domain=%s payload=%s",
                escalation_type,
                correlation_id,
                payload.get("user_id"),
                payload.get("domain"),
                json.dumps(payload),
                exc_info=True,
            )
            raise


# ── Public interface ───────────────────────────────────────────────────────────

def is_escalation_configured() -> bool:
    """True if at least one escalation channel is ready."""
    return _zendesk_configured() or _sb_configured()


def raise_ticket(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
    conversation_reference: dict | None = None,
    user_email: str | None = None,
) -> str:
    """
    Create a support ticket.
    Returns a reference string: "ZD-{id}" via Zendesk, "REF-{hex}" via SB.
    Raises RuntimeError if no escalation channel is configured.
    """
    correlation_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    if _zendesk_configured():
        subject = f"[{domain.upper()}] Support request from Teams — {question_text[:60]}"
        body = (
            f"**Domain:** {domain}\n"
            f"**Question:** {question_text}\n\n"
            f"**Conversation ID:** {conversation_id}\n"
            f"**User ID:** {user_id}\n"
            f"**Question ID:** {question_id}\n"
            f"**Timestamp:** {timestamp}\n"
        )
        try:
            ref = _zendesk_create_ticket(
                subject=subject,
                body=body,
                tags=["rag-bot", f"domain:{domain.lower()}", "raise_ticket"],
                external_id=correlation_id,
                group_id=settings.ZENDESK_GROUP_ID_TICKET,
                requester_email=user_email,
            )
            logger.info(
                "escalation_ticket_created via=zendesk ref=%s user_id=%s domain=%s",
                ref, user_id, domain,
            )
            return ref
        except Exception as exc:
            logger.error(
                "zendesk_ticket_failed correlation_id=%s user_id=%s domain=%s: %s",
                correlation_id, user_id, domain, exc, exc_info=True,
            )
            if not _sb_configured():
                raise RuntimeError(f"Ticket creation failed and no fallback configured: {exc}") from exc
            logger.warning("escalation_fallback_to_servicebus correlation_id=%s", correlation_id)

    if _sb_configured():
        payload = {
            "type":                   "raise_ticket",
            "correlation_id":         correlation_id,
            "user_id":                user_id,
            "conversation_id":        conversation_id,
            "question_id":            question_id,
            "question_text":          question_text[:1000],
            "domain":                 domain,
            "timestamp":              timestamp,
            "conversation_reference": conversation_reference or {},
        }
        _sb_send("raise_ticket", payload, correlation_id)
        logger.info(
            "escalation_ticket_queued via=service_bus correlation_id=%s user_id=%s domain=%s",
            correlation_id, user_id, domain,
        )
        return correlation_id

    raise RuntimeError(
        "No escalation channel configured. "
        "Set ZENDESK_SUBDOMAIN/ZENDESK_API_TOKEN/ZENDESK_USER_EMAIL "
        "or AZURE_SERVICE_BUS_NAMESPACE."
    )


def connect_sme(
    user_id: str,
    conversation_id: str,
    question_id: str,
    question_text: str,
    domain: str,
    conversation_reference: dict | None = None,
    user_email: str | None = None,
) -> str:
    """
    Request SME connection via a Zendesk ticket (or Service Bus fallback).
    Returns a reference string.
    Raises RuntimeError if no escalation channel is configured.
    """
    correlation_id = f"REF-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    if _zendesk_configured():
        subject = f"[{domain.upper()}] SME connection request — {question_text[:60]}"
        body = (
            f"**SME Connection Request**\n\n"
            f"**Domain:** {domain}\n"
            f"**Question:** {question_text}\n\n"
            f"**Conversation ID:** {conversation_id}\n"
            f"**User ID:** {user_id}\n"
            f"**Question ID:** {question_id}\n"
            f"**Timestamp:** {timestamp}\n\n"
            f"*Please connect the user with a subject matter expert for this query.*"
        )
        try:
            ref = _zendesk_create_ticket(
                subject=subject,
                body=body,
                tags=["rag-bot", f"domain:{domain.lower()}", "connect_sme", "sme-request"],
                external_id=correlation_id,
                group_id=settings.ZENDESK_GROUP_ID_SME,
                requester_email=user_email,
            )
            logger.info(
                "escalation_sme_created via=zendesk ref=%s user_id=%s domain=%s",
                ref, user_id, domain,
            )
            return ref
        except Exception as exc:
            logger.error(
                "zendesk_sme_failed correlation_id=%s user_id=%s domain=%s: %s",
                correlation_id, user_id, domain, exc, exc_info=True,
            )
            if not _sb_configured():
                raise RuntimeError(f"SME connection failed and no fallback configured: {exc}") from exc
            logger.warning("escalation_fallback_to_servicebus correlation_id=%s", correlation_id)

    if _sb_configured():
        payload = {
            "type":                   "connect_sme",
            "correlation_id":         correlation_id,
            "user_id":                user_id,
            "conversation_id":        conversation_id,
            "question_id":            question_id,
            "question_text":          question_text[:1000],
            "domain":                 domain,
            "timestamp":              timestamp,
            "conversation_reference": conversation_reference or {},
        }
        _sb_send("connect_sme", payload, correlation_id)
        logger.info(
            "escalation_sme_queued via=service_bus correlation_id=%s user_id=%s domain=%s",
            correlation_id, user_id, domain,
        )
        return correlation_id

    raise RuntimeError(
        "No escalation channel configured. "
        "Set ZENDESK_SUBDOMAIN/ZENDESK_API_TOKEN/ZENDESK_USER_EMAIL "
        "or AZURE_SERVICE_BUS_NAMESPACE."
    )
