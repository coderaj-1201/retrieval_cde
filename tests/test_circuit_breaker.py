"""Unit tests for shared/circuit_breaker.py — no Azure dependencies."""
from __future__ import annotations

import asyncio
import time
import pytest

from shared.circuit_breaker import CircuitBreaker, CircuitOpenError


async def _ok():
    return "ok"


async def _fail():
    raise ValueError("simulated failure")


@pytest.mark.asyncio
async def test_closed_state_passes_calls():
    cb = CircuitBreaker(name="test", fail_max=3, reset_timeout=30)
    result = await cb.call(_ok)
    assert result == "ok"
    assert cb.to_dict()["state"] == "closed"


@pytest.mark.asyncio
async def test_opens_after_fail_max():
    cb = CircuitBreaker(name="test", fail_max=3, reset_timeout=30)
    for _ in range(3):
        with pytest.raises(ValueError):
            await cb.call(_fail)
    assert cb.to_dict()["state"] == "open"


@pytest.mark.asyncio
async def test_open_state_raises_circuit_open_error():
    cb = CircuitBreaker(name="test", fail_max=2, reset_timeout=30)
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(_fail)

    with pytest.raises(CircuitOpenError) as exc_info:
        await cb.call(_ok)
    assert exc_info.value.retry_after > 0


@pytest.mark.asyncio
async def test_half_open_after_reset_timeout():
    cb = CircuitBreaker(name="test", fail_max=2, reset_timeout=0.05)
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(_fail)

    assert cb.to_dict()["state"] == "open"
    await asyncio.sleep(0.1)

    # After reset_timeout, next call enters HALF_OPEN and succeeds → CLOSED
    result = await cb.call(_ok)
    assert result == "ok"
    assert cb.to_dict()["state"] == "closed"


@pytest.mark.asyncio
async def test_half_open_failure_reopens():
    cb = CircuitBreaker(name="test", fail_max=2, reset_timeout=0.05)
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(_fail)

    await asyncio.sleep(0.1)

    with pytest.raises(ValueError):
        await cb.call(_fail)
    assert cb.to_dict()["state"] == "open"


def test_to_dict_structure():
    cb = CircuitBreaker(name="my-service", fail_max=5, reset_timeout=60)
    d = cb.to_dict()
    assert d["name"]         == "my-service"
    assert d["state"]        == "closed"
    assert d["fail_count"]   == 0
    assert "opened_at"       in d
