"""
Adversarial tests for Zendesk escalation path and telemetry meter initialisation.

"If the ticket doesn't get raised, the user is stuck. This must not fail silently."
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch, call
import pytest
import httpx

# ── Stub agent_framework ───────────────────────────────────────────────────────
_af_stub = MagicMock()
_af_stub.step     = lambda fn: fn
_af_stub.workflow = MagicMock(return_value=lambda fn: fn)
sys.modules.setdefault("agent_framework", _af_stub)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ZENDESK TICKET CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def _zendesk_settings(
    subdomain="myco",
    token="zk_token",
    email="bot@myco.com",
    group_ticket=None,
    group_sme=None,
):
    ms = MagicMock()
    ms.ZENDESK_SUBDOMAIN       = subdomain
    ms.ZENDESK_API_TOKEN       = MagicMock()
    ms.ZENDESK_API_TOKEN.get_secret_value.return_value = token
    ms.ZENDESK_USER_EMAIL      = email
    ms.ZENDESK_GROUP_ID_TICKET = group_ticket
    ms.ZENDESK_GROUP_ID_SME    = group_sme
    ms.AZURE_SERVICE_BUS_CONNECTION_STR = None
    ms.AZURE_SERVICE_BUS_NAMESPACE      = None
    ms.SB_QUEUE_ESCALATION              = "escalation-requests"
    return ms


class TestZendeskRaiseTicket:

    def _mock_response(self, ticket_id=12345, status_code=201):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = {"ticket": {"id": ticket_id}}
        resp.raise_for_status = MagicMock()   # no-op on success
        return resp

    def test_raise_ticket_returns_zendesk_ref(self):
        """Successful ticket creation returns 'ZD-{ticket_id}'."""
        from shared.escalation_client import raise_ticket

        mock_resp = self._mock_response(ticket_id=9001)

        with patch("shared.escalation_client.settings", _zendesk_settings()), \
             patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_resp
            ref = raise_ticket(
                user_id="u-1", conversation_id="c-1",
                question_id="q-1", question_text="What is my leave balance?",
                domain="hr",
            )

        assert ref == "ZD-9001"

    def test_raise_ticket_sends_correct_payload(self):
        """The Zendesk API call must include subject, body, tags, and external_id."""
        from shared.escalation_client import raise_ticket

        mock_resp = self._mock_response(ticket_id=1)
        captured = {}

        def _fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json", {})
            return mock_resp

        with patch("shared.escalation_client.settings", _zendesk_settings(subdomain="acme")), \
             patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = _fake_post
            raise_ticket(
                user_id="u-1", conversation_id="c-1", question_id="q-1",
                question_text="How do I reset my password?", domain="it",
            )

        assert "acme.zendesk.com" in captured["url"]
        ticket = captured["json"]["ticket"]
        assert "it" in ticket["subject"].lower() or "IT" in ticket["subject"]
        assert "raise_ticket" in ticket["tags"]
        assert "domain:it" in ticket["tags"]
        assert ticket["external_id"].startswith("REF-")

    def test_raise_ticket_with_group_id(self):
        """Group ID is included in the payload when configured."""
        from shared.escalation_client import raise_ticket

        mock_resp = self._mock_response(ticket_id=42)
        captured_ticket = {}

        def _fake_post(url, **kwargs):
            captured_ticket.update(kwargs.get("json", {}).get("ticket", {}))
            return mock_resp

        settings = _zendesk_settings(group_ticket=777)

        with patch("shared.escalation_client.settings", settings), \
             patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = _fake_post
            raise_ticket(
                user_id="u", conversation_id="c", question_id="q",
                question_text="Q", domain="ops",
            )

        assert captured_ticket.get("group_id") == 777

    def test_raise_ticket_with_requester_email(self):
        """User email is forwarded as the Zendesk ticket requester."""
        from shared.escalation_client import raise_ticket

        mock_resp = self._mock_response(ticket_id=55)
        captured_ticket = {}

        def _fake_post(url, **kwargs):
            captured_ticket.update(kwargs.get("json", {}).get("ticket", {}))
            return mock_resp

        with patch("shared.escalation_client.settings", _zendesk_settings()), \
             patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = _fake_post
            raise_ticket(
                user_id="u", conversation_id="c", question_id="q",
                question_text="Q", domain="hr",
                user_email="alice@corp.com",
            )

        assert captured_ticket.get("requester", {}).get("email") == "alice@corp.com"

    def test_raise_ticket_zendesk_http_error_raises_runtime(self):
        """A non-2xx Zendesk response becomes RuntimeError."""
        from shared.escalation_client import raise_ticket

        err_resp = MagicMock()
        err_resp.status_code = 422
        err_resp.text = "Invalid ticket format"
        err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "422", request=MagicMock(), response=err_resp
        )

        with patch("shared.escalation_client.settings", _zendesk_settings()), \
             patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = err_resp
            with pytest.raises(RuntimeError, match="Zendesk API error 422"):
                raise_ticket(
                    user_id="u", conversation_id="c", question_id="q",
                    question_text="Q", domain="hr",
                )

    def test_raise_ticket_zendesk_network_error_raises_runtime(self):
        """Connection error becomes RuntimeError."""
        from shared.escalation_client import raise_ticket

        with patch("shared.escalation_client.settings", _zendesk_settings()), \
             patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = \
                httpx.ConnectError("Connection refused")
            with pytest.raises(RuntimeError, match="Zendesk request failed"):
                raise_ticket(
                    user_id="u", conversation_id="c", question_id="q",
                    question_text="Q", domain="hr",
                )


class TestZendeskConnectSME:

    def test_connect_sme_returns_zendesk_ref(self):
        """SME ticket created via Zendesk returns 'ZD-{id}'."""
        from shared.escalation_client import connect_sme

        resp = MagicMock()
        resp.json.return_value = {"ticket": {"id": 77}}
        resp.raise_for_status = MagicMock()

        with patch("shared.escalation_client.settings", _zendesk_settings(group_sme=888)), \
             patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = resp
            ref = connect_sme(
                user_id="u", conversation_id="c", question_id="q",
                question_text="Need an expert on trade finance.", domain="legal",
            )

        assert ref == "ZD-77"

    def test_connect_sme_tags_include_sme_request(self):
        """SME tickets must carry the 'connect_sme' and 'sme-request' tags."""
        from shared.escalation_client import connect_sme

        captured = {}

        def _fake_post(url, **kwargs):
            captured.update(kwargs.get("json", {}).get("ticket", {}))
            r = MagicMock()
            r.json.return_value = {"ticket": {"id": 1}}
            r.raise_for_status = MagicMock()
            return r

        with patch("shared.escalation_client.settings", _zendesk_settings()), \
             patch("httpx.Client") as mc:
            mc.return_value.__enter__.return_value.post.side_effect = _fake_post
            connect_sme(
                user_id="u", conversation_id="c", question_id="q",
                question_text="Q", domain="ops",
            )

        assert "connect_sme"  in captured.get("tags", [])
        assert "sme-request"  in captured.get("tags", [])


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FALLBACK TO SERVICE BUS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEscalationFallback:

    def _sb_settings(self):
        ms = MagicMock()
        ms.ZENDESK_SUBDOMAIN  = None   # Zendesk not configured
        ms.ZENDESK_API_TOKEN  = None
        ms.ZENDESK_USER_EMAIL = None
        ms.AZURE_SERVICE_BUS_NAMESPACE      = "myns.servicebus.windows.net"
        ms.AZURE_SERVICE_BUS_CONNECTION_STR = None
        ms.SB_QUEUE_ESCALATION = "escalation-requests"
        return ms

    def test_falls_back_to_sb_when_zendesk_not_configured(self):
        """When Zendesk is absent, SB path is taken and a REF- string returned."""
        from shared.escalation_client import raise_ticket

        mock_sender = MagicMock()
        mock_sender.__enter__ = MagicMock(return_value=mock_sender)
        mock_sender.__exit__ = MagicMock(return_value=False)

        with patch("shared.escalation_client.settings", self._sb_settings()), \
             patch("shared.escalation_client._sb_get_sender", return_value=mock_sender):
            ref = raise_ticket(
                user_id="u", conversation_id="c", question_id="q",
                question_text="Q", domain="hr",
            )

        assert ref.startswith("REF-")
        mock_sender.send_messages.assert_called_once()

    def test_zendesk_failure_falls_back_to_sb(self):
        """Zendesk HTTP error falls back to SB when SB is also configured."""
        from shared.escalation_client import raise_ticket

        ms = _zendesk_settings()
        ms.AZURE_SERVICE_BUS_NAMESPACE = "myns.servicebus.windows.net"

        err_resp = MagicMock()
        err_resp.status_code = 500
        err_resp.text = "Internal Server Error"
        err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=err_resp
        )

        mock_sender = MagicMock()
        mock_sender.__enter__ = MagicMock(return_value=mock_sender)
        mock_sender.__exit__ = MagicMock(return_value=False)

        with patch("shared.escalation_client.settings", ms), \
             patch("httpx.Client") as mock_client, \
             patch("shared.escalation_client._sb_get_sender", return_value=mock_sender):
            mock_client.return_value.__enter__.return_value.post.return_value = err_resp
            ref = raise_ticket(
                user_id="u", conversation_id="c", question_id="q",
                question_text="Q", domain="hr",
            )

        assert ref.startswith("REF-")
        mock_sender.send_messages.assert_called_once()

    def test_no_channel_raises_runtime_error(self):
        """If neither Zendesk nor SB is configured, RuntimeError is raised."""
        from shared.escalation_client import raise_ticket

        ms = MagicMock()
        ms.ZENDESK_SUBDOMAIN  = None
        ms.ZENDESK_API_TOKEN  = None
        ms.ZENDESK_USER_EMAIL = None
        ms.AZURE_SERVICE_BUS_NAMESPACE      = None
        ms.AZURE_SERVICE_BUS_CONNECTION_STR = None

        with patch("shared.escalation_client.settings", ms):
            with pytest.raises(RuntimeError, match="No escalation channel"):
                raise_ticket(
                    user_id="u", conversation_id="c", question_id="q",
                    question_text="Q", domain="hr",
                )

    def test_is_escalation_configured_zendesk(self):
        """is_escalation_configured returns True when Zendesk is set."""
        from shared.escalation_client import is_escalation_configured
        with patch("shared.escalation_client.settings", _zendesk_settings()):
            assert is_escalation_configured() is True

    def test_is_escalation_configured_neither(self):
        """is_escalation_configured returns False when nothing is set."""
        from shared.escalation_client import is_escalation_configured
        ms = MagicMock()
        ms.ZENDESK_SUBDOMAIN  = None
        ms.ZENDESK_API_TOKEN  = None
        ms.ZENDESK_USER_EMAIL = None
        ms.AZURE_SERVICE_BUS_NAMESPACE      = None
        ms.AZURE_SERVICE_BUS_CONNECTION_STR = None
        with patch("shared.escalation_client.settings", ms):
            assert is_escalation_configured() is False


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TELEMETRY METER INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestTelemetrySetup:

    def test_recording_before_setup_is_silent_noop(self):
        """Record functions must not raise even before setup_meters() is called."""
        import importlib
        import shared.telemetry as tel

        # Temporarily clear instruments to simulate pre-setup state
        saved = tel._query_counter
        tel._query_counter = None
        try:
            tel.record_query(domain="hr", status="success", tool="hybrid")  # must not raise
        finally:
            tel._query_counter = saved

    def test_setup_meters_binds_instruments(self):
        """After setup_meters(), instruments are non-None."""
        import shared.telemetry as tel

        # Reset for clean test
        saved = {k: getattr(tel, k) for k in
                 ["_query_counter", "_confidence_histo", "_attempts_histo",
                  "_escalation_counter", "_tool_counter"]}
        for k in saved:
            setattr(tel, k, None)

        try:
            tel.setup_meters()
            assert tel._query_counter      is not None
            assert tel._confidence_histo   is not None
            assert tel._attempts_histo     is not None
            assert tel._escalation_counter is not None
            assert tel._tool_counter       is not None
        finally:
            for k, v in saved.items():
                setattr(tel, k, v)

    def test_setup_meters_idempotent(self):
        """Calling setup_meters() twice does not raise or reset instruments."""
        import shared.telemetry as tel
        tel.setup_meters()
        first = tel._query_counter
        tel.setup_meters()   # second call
        assert tel._query_counter is first   # same object — not recreated

    def test_record_all_metrics_no_raise(self):
        """All recording functions must be callable without error after setup."""
        import shared.telemetry as tel
        tel.setup_meters()

        tel.record_query(domain="ops", status="success", tool="hybrid")
        tel.record_confidence(confidence=0.87, domain="ops", status="success")
        tel.record_attempts(attempts=2, domain="ops", status="success")
        tel.record_escalation(escalation_type="raise_ticket", domain="hr")
        tel.record_tool(tool="hyde", domain="legal")
