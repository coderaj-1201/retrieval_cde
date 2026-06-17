"""
Teams Bot Adapter
=================
FastAPI/uvicorn server bridging Microsoft Teams and the Main Agent.

Flow:
  Teams → POST /api/messages → Activity → MAF @step → Main Agent → Adaptive Card → Teams

Security notes:
  - /test-message and /test-feedback are only registered in non-production environments.
    They are completely absent (404) in ENVIRONMENT=production.
  - All user text is length-validated and control-character stripped before forwarding.
  - Payload size is capped at 1 MB before the request body is read.
"""
from __future__ import annotations

import json
import logging
import os
import hashlib
import uuid
import re
from contextlib import asynccontextmanager

import httpx
import uvicorn
from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.integration.aiohttp import CloudAdapter, ConfigurationBotFrameworkAuthentication
from botbuilder.schema import Activity, ActivityTypes, Attachment
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from card_mapper import build_answer_card, build_feedback_card

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Env vars ───────────────────────────────────────────────────────────────────
MICROSOFT_APP_ID        = os.getenv("MicrosoftAppId", "")
MICROSOFT_APP_PASSWORD  = os.getenv("MicrosoftAppPassword", "")
MICROSOFT_APP_TYPE      = os.getenv("MicrosoftAppType", "MultiTenant")
MICROSOFT_APP_TENANT_ID = os.getenv("MicrosoftAppTenantId", "")
MAIN_AGENT_URL          = os.getenv("MAIN_AGENT_URL", "http://localhost:8000")
BOT_PORT                = int(os.getenv("BOT_PORT", 3978))
ENVIRONMENT             = os.getenv("ENVIRONMENT", "production").lower()

# Maximum characters accepted from a user message after mention removal.
# Matches MAX_QUERY_LENGTH on the main agent side.
_MAX_USER_TEXT = int(os.getenv("MAX_QUERY_LENGTH", "2000"))


# ── Bot Framework adapter ──────────────────────────────────────────────────────

class _BotConfig:
    APP_ID       = MICROSOFT_APP_ID
    APP_PASSWORD = MICROSOFT_APP_PASSWORD
    APP_TYPE     = MICROSOFT_APP_TYPE
    APP_TENANTID = MICROSOFT_APP_TENANT_ID


ADAPTER = CloudAdapter(ConfigurationBotFrameworkAuthentication(_BotConfig()))


async def on_error(context: TurnContext, error: Exception) -> None:
    logger.error("[on_turn_error] %s", error, exc_info=True)
    await context.send_activity(
        "The bot encountered an error while processing your request. Please try again."
    )


ADAPTER.on_turn_error = on_error


# ── Payload size middleware ────────────────────────────────────────────────────

class _ContentSizeLimitMiddleware(BaseHTTPMiddleware):
    _MAX_BYTES = 1_048_576  # 1 MB

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self._MAX_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": "payload_too_large", "max_bytes": self._MAX_BYTES},
            )
        return await call_next(request)


# ── Downstream helpers ─────────────────────────────────────────────────────────

async def call_main_agent(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{MAIN_AGENT_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


async def call_feedback(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{MAIN_AGENT_URL}/feedback",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_tenant_id(turn_context: TurnContext) -> str | None:
    activity = turn_context.activity
    if activity.channel_data:
        tenant = activity.channel_data.get("tenant", {})
        if tenant and tenant.get("id"):
            return tenant.get("id")
    if activity.conversation and hasattr(activity.conversation, "tenant_id"):
        return activity.conversation.tenant_id
    return None


def remove_bot_mention(turn_context: TurnContext, text: str) -> str:
    """
    Strip the bot's @mention from the message text.
    Uses the bot's AAD object ID for matching (not display name) so this
    is robust to bot renames in the Teams manifest.
    """
    activity = turn_context.activity
    if not text:
        return ""
    if not activity.entities:
        return text.strip()

    bot_id = activity.recipient.id if activity.recipient else None

    for entity in activity.entities:
        if entity.type == "mention":
            mentioned    = entity.additional_properties.get("mentioned", {})
            mentioned_id = mentioned.get("id")
            if bot_id and mentioned_id == bot_id:
                mention_text = entity.additional_properties.get("text", "")
                text = text.replace(mention_text, "")
    return text.strip()


# Common profanity patterns — replace with asterisks before storage.
# This is a lightweight policy-safe filter; extend the list as needed.
_PROFANITY_RE = re.compile(
    r"\b(fuck|shit|ass(?:hole)?|bitch|bastard|cunt|dick|piss|crap|damn|hell)\b",
    re.IGNORECASE,
)


def _redact_profanity(text: str) -> str:
    """Replace matched profanity words with asterisks of the same length."""
    return _PROFANITY_RE.sub(lambda m: "*" * len(m.group()), text)


def _sanitise_user_text(text: str) -> str:
    """Strip control characters, redact profanity, enforce length cap."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()
    text = _redact_profanity(text)
    if len(text) > _MAX_USER_TEXT:
        logger.warning(
            "teams_message_truncated original_len=%d max_len=%d",
            len(text), _MAX_USER_TEXT,
        )
        text = text[:_MAX_USER_TEXT]
    return text


# ── Bot handler ────────────────────────────────────────────────────────────────

class IronmanBot(ActivityHandler):

    async def on_message_activity(self, turn_context: TurnContext) -> None:
        if turn_context.activity.value and isinstance(turn_context.activity.value, dict):
            await self._handle_card_action(turn_context)
            return
        await self._handle_user_question(turn_context)

    async def _handle_user_question(self, turn_context: TurnContext) -> None:
        activity = turn_context.activity
        await turn_context.send_activity(Activity(type=ActivityTypes.typing))

        raw_text  = remove_bot_mention(turn_context, activity.text or "")
        user_text = _sanitise_user_text(raw_text)

        if not user_text:
            await turn_context.send_activity("Please send a message.")
            return

        from_prop       = activity.from_property
        user_id         = (
            getattr(from_prop, "aad_object_id", None)
            or getattr(from_prop, "aadObjectId", None)
            or (from_prop.id if from_prop else None)
            or "anonymous"
        )
        conversation_id = (
            activity.conversation.id if activity.conversation else str(uuid.uuid4())
        )
        tenant_id = get_tenant_id(turn_context)

        logger.info(
            "teams_message user_id=%s tenant=%s conversation=%.40s text=%.80s",
            user_id, tenant_id, conversation_id, user_text,
        )

        # Stable idempotency key: same user + same conversation + same text
        # always maps to the same key, so repeated sends return the cached answer.
        idempotency_key = "q-" + hashlib.sha256(
            f"{user_id}:{conversation_id}:{user_text}".encode()
        ).hexdigest()[:24]

        try:
            data = await call_main_agent({
                "text":             user_text,
                "conversation_id":  conversation_id,
                "user_id":          user_id,
                "idempotency_key":  idempotency_key,
            })
        except Exception as exc:
            logger.error("main_agent_call_failed: %s", exc, exc_info=True)
            await turn_context.send_activity(
                "⚠️ Service temporarily unavailable. Please try again."
            )
            return

        answer_text = data.get("answer", "").strip()
        msg_status  = data.get("status")

        if msg_status in ("ticket_raised", "sme_connecting"):
            await turn_context.send_activity(
                answer_text or "Your request has been escalated."
            )
            return

        if answer_text:
            answer_card = build_answer_card(data)
            await turn_context.send_activity(Activity(
                type=ActivityTypes.message,
                attachments=[Attachment(
                    content_type=answer_card["contentType"],
                    content=answer_card["content"],
                )],
            ))
            if msg_status == "success":
                feedback_card = build_feedback_card(data)
                await turn_context.send_activity(Activity(
                    type=ActivityTypes.message,
                    attachments=[Attachment(
                        content_type=feedback_card["contentType"],
                        content=feedback_card["content"],
                    )],
                ))
        else:
            await turn_context.send_activity(
                "⚠️ I wasn't able to find an answer. Please try rephrasing your question."
            )

    async def _handle_card_action(self, turn_context: TurnContext) -> None:
        value  = turn_context.activity.value or {}
        action = value.get("action")
        if action == "feedback":
            await self._handle_feedback(turn_context, value)
        elif action == "escalate":
            await self._handle_escalate(turn_context, value)
        else:
            await turn_context.send_activity("Action received.")

    async def _handle_feedback(self, turn_context: TurnContext, value: dict) -> None:
        from_prop = turn_context.activity.from_property
        raw    = value.get("feedback", "")
        rating = (
            "thumbs_up" if raw == "positive"
            else "thumbs_down" if raw == "negative"
            else "neutral"
        )
        comment = _sanitise_user_text(value.get("feedback_comment", ""))[:2000]

        payload = {
            "question_id":     value.get("question_id", ""),
            "answer_id":       value.get("answer_id", ""),
            "conversation_id": value.get("conversation_id", ""),
            "user_id":         value.get("user_id") or (from_prop.id if from_prop else "anonymous"),
            "rating":          rating,
            "comment":         comment,
        }
        try:
            await call_feedback(payload)
            msg = (
                "👍 Thanks, glad it was helpful!"
                if rating == "thumbs_up"
                else "👎 Thanks, we'll use this to improve!"
            )
            thanks_card = {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type":    "AdaptiveCard",
                "version": "1.4",
                "body": [{
                    "type": "TextBlock", "text": msg,
                    "wrap": True, "size": "Small", "isSubtle": True,
                }],
            }
            reply_id = turn_context.activity.reply_to_id
            if reply_id:
                from botbuilder.schema import Activity as _A, ActivityTypes as _AT, Attachment as _Att
                await turn_context.update_activity(_A(
                    id=reply_id,
                    type=_AT.message,
                    attachments=[_Att(
                        content_type="application/vnd.microsoft.card.adaptive",
                        content=thanks_card,
                    )],
                ))
            else:
                await turn_context.send_activity(msg)

            logger.info(
                "feedback_saved question_id=%s rating=%s",
                payload["question_id"], rating,
            )
        except Exception as exc:
            logger.error("feedback_failed: %s", exc, exc_info=True)
            await turn_context.send_activity("Couldn't save feedback. Please try again.")

    async def _handle_escalate(self, turn_context: TurnContext, value: dict) -> None:
        escalation_type = value.get("escalation_type", "raise_ticket")
        from_prop       = turn_context.activity.from_property
        user_id         = from_prop.id if from_prop else "anonymous"
        conv_id         = turn_context.activity.conversation.id
        try:
            data = await call_main_agent({
                "text":            escalation_type,
                "conversation_id": conv_id,
                "user_id":         user_id,
            })
            await turn_context.send_activity(
                data.get("answer", "Escalation request received.")
            )
            # Show a lightweight feedback card so users can still rate the
            # escalation outcome even though no answer card was produced.
            if data.get("answer_id"):
                fb_card = build_feedback_card(data)
                await turn_context.send_activity(Activity(
                    type=ActivityTypes.message,
                    attachments=[Attachment(
                        content_type=fb_card["contentType"],
                        content=fb_card["content"],
                    )],
                ))
        except Exception as exc:
            logger.error("escalate_failed: %s", exc, exc_info=True)
            await turn_context.send_activity(
                "Couldn't process escalation. Please try again."
            )

    async def on_members_added_activity(self, members_added, turn_context: TurnContext) -> None:
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hi, I am IRONMAN ChatBot! 👋")


BOT = IronmanBot()


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "teams_bot_started port=%d environment=%s app_id=%s main_agent=%s",
        BOT_PORT, ENVIRONMENT,
        MICROSOFT_APP_ID or "(local dev — no auth)", MAIN_AGENT_URL,
    )
    if ENVIRONMENT == "production" and not MICROSOFT_APP_ID:
        logger.warning(
            "teams_bot_no_app_id: MicrosoftAppId is not set in production. "
            "Bot Framework authentication will fail."
        )
    yield
    logger.info("teams_bot_stopped")


app = FastAPI(title="IRONMAN Teams Bot", lifespan=lifespan)
app.add_middleware(_ContentSizeLimitMiddleware)


@app.get("/health/live")
async def liveness() -> dict:
    return {"status": "alive", "agent": "teams-bot"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "agent": "teams-bot"}


@app.post("/api/messages")
async def messages(request: Request) -> Response:
    if "application/json" not in request.headers.get("content-type", ""):
        return Response(status_code=415)

    body        = await request.json()
    activity    = Activity().deserialize(body)
    auth_header = request.headers.get("Authorization", "")

    async def turn_handler(turn_context: TurnContext) -> None:
        await BOT.on_turn(turn_context)

    try:
        invoke_response = await ADAPTER.process_activity(auth_header, activity, turn_handler)
        if invoke_response:
            return JSONResponse(
                status_code=invoke_response.status,
                content=invoke_response.body,
            )
        return Response(status_code=201)
    except PermissionError as exc:
        logger.error("bot_auth_error: %s", exc)
        return Response(status_code=401)
    except Exception as exc:
        logger.error("bot_adapter_error: %s", exc, exc_info=True)
        return Response(status_code=500)


# ── Development-only endpoints — NOT registered in production ──────────────────
# These are absent (404) when ENVIRONMENT=production, which is the ACA default.

if ENVIRONMENT != "production":
    logger.warning(
        "teams_bot_dev_endpoints_enabled environment=%s — "
        "/test-message and /test-feedback are active. "
        "These must NOT be reachable in production.",
        ENVIRONMENT,
    )

    @app.post("/test-message")
    async def test_message(request: Request) -> Response:
        """Dev only — simulate a Teams message without Bot Framework auth."""
        body    = await request.json()
        text    = body.get("text", "")
        user_id = body.get("user_id", "test-user")
        conv_id = body.get("conversation_id", str(uuid.uuid4()))
        try:
            data = await call_main_agent({
                "text": _sanitise_user_text(text),
                "conversation_id": conv_id,
                "user_id": user_id,
            })
        except Exception as exc:
            return Response(
                content=json.dumps({"error": str(exc)}),
                media_type="application/json",
                status_code=500,
            )
        return Response(
            content=json.dumps({
                "reply_card":    build_answer_card(data),
                "raw":           data,
                "conversation_id": conv_id,
            }),
            media_type="application/json",
        )

    @app.post("/test-feedback")
    async def test_feedback(request: Request) -> Response:
        """Dev only — simulate feedback button click."""
        body = await request.json()
        try:
            result = await call_feedback(body)
            return Response(content=json.dumps(result), media_type="application/json")
        except Exception as exc:
            return Response(
                content=json.dumps({"error": str(exc)}),
                media_type="application/json",
                status_code=500,
            )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=BOT_PORT, reload=False)
