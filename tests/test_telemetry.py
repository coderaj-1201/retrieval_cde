"""Unit tests for shared/telemetry.py — verifies no-op safety when OTel is unavailable."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


def test_record_query_no_op_when_metrics_unavailable():
    """All recording functions must be silent when _METRICS_AVAILABLE is False."""
    import shared.telemetry as tel
    original = tel._METRICS_AVAILABLE
    try:
        tel._METRICS_AVAILABLE = False
        tel.record_query(domain="ops", status="success", tool="hybrid")
        tel.record_confidence(confidence=0.85, domain="ops", status="success")
        tel.record_attempts(attempts=1, domain="ops", status="success")
        tel.record_escalation(escalation_type="raise_ticket", domain="hr")
        tel.record_tool(tool="hyde", domain="it")
        # No exception = pass
    finally:
        tel._METRICS_AVAILABLE = original


def test_record_query_with_mock_counter():
    import shared.telemetry as tel
    original = tel._METRICS_AVAILABLE
    try:
        tel._METRICS_AVAILABLE = True
        mock_counter = MagicMock()
        with patch.object(tel, "_query_counter", mock_counter):
            tel.record_query(domain="hr", status="success", tool="hybrid")
        mock_counter.add.assert_called_once_with(
            1, {"domain": "hr", "status": "success", "tool": "hybrid"}
        )
    finally:
        tel._METRICS_AVAILABLE = original


def test_record_confidence_clamps_value():
    import shared.telemetry as tel
    original = tel._METRICS_AVAILABLE
    try:
        tel._METRICS_AVAILABLE = True
        mock_hist = MagicMock()
        with patch.object(tel, "_confidence_histogram", mock_hist):
            tel.record_confidence(confidence=1.5, domain="ops", status="success")
        args = mock_hist.record.call_args[0]
        assert args[0] == 1.0   # clamped to [0, 1]

        with patch.object(tel, "_confidence_histogram", mock_hist):
            tel.record_confidence(confidence=-0.3, domain="ops", status="failure")
        args = mock_hist.record.call_args[0]
        assert args[0] == 0.0
    finally:
        tel._METRICS_AVAILABLE = original


def test_record_escalation_with_mock():
    import shared.telemetry as tel
    original = tel._METRICS_AVAILABLE
    try:
        tel._METRICS_AVAILABLE = True
        mock_counter = MagicMock()
        with patch.object(tel, "_escalation_counter", mock_counter):
            tel.record_escalation(escalation_type="connect_sme", domain="legal")
        mock_counter.add.assert_called_once_with(
            1, {"type": "connect_sme", "domain": "legal"}
        )
    finally:
        tel._METRICS_AVAILABLE = original


def test_recording_swallows_otel_errors():
    """A broken OTel exporter must not crash the application."""
    import shared.telemetry as tel
    original = tel._METRICS_AVAILABLE
    try:
        tel._METRICS_AVAILABLE = True
        mock_counter = MagicMock()
        mock_counter.add.side_effect = RuntimeError("exporter down")
        with patch.object(tel, "_query_counter", mock_counter):
            tel.record_query(domain="it", status="error")   # must not raise
    finally:
        tel._METRICS_AVAILABLE = original
