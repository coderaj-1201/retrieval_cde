"""
Async-native circuit breaker for inter-agent HTTP calls.

States:
  CLOSED   — normal operation, requests pass through.
  OPEN     — failure threshold exceeded, requests fast-fail for reset_timeout seconds.
  HALF_OPEN — one probe request is allowed to test if the downstream has recovered.

Usage:
    from shared.circuit_breaker import CircuitBreaker, CircuitOpenError

    _breaker = CircuitBreaker(name="retrieval", fail_max=3, reset_timeout=30)

    try:
        result = await _breaker.call(my_async_fn, *args)
    except CircuitOpenError:
        # return fast failure response
"""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """Raised when the circuit is OPEN and a call is rejected."""
    def __init__(self, name: str, retry_after: float) -> None:
        self.name        = name
        self.retry_after = round(retry_after, 1)
        super().__init__(
            f"Circuit '{name}' is OPEN. Retry after {self.retry_after}s."
        )


class _State(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Thread-safe (asyncio.Lock) circuit breaker.

    Args:
        name          : identifier used in logs.
        fail_max      : consecutive failures before opening the circuit.
        reset_timeout : seconds the circuit stays open before allowing a probe.
    """

    def __init__(self, name: str, fail_max: int = 3, reset_timeout: float = 30.0) -> None:
        self.name          = name
        self.fail_max      = fail_max
        self.reset_timeout = reset_timeout
        self._state        = _State.CLOSED
        self._fail_count   = 0
        self._opened_at: float | None = None
        self._probing      = False   # True while a HALF_OPEN probe is in-flight
        self._lock         = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state.value

    async def call(self, fn, *args, **kwargs):
        """
        Invoke `fn(*args, **kwargs)` through the circuit breaker.
        Raises CircuitOpenError immediately if the circuit is OPEN.
        """
        async with self._lock:
            if self._state == _State.OPEN:
                elapsed = time.monotonic() - (self._opened_at or 0)
                if elapsed < self.reset_timeout:
                    raise CircuitOpenError(
                        self.name,
                        retry_after=self.reset_timeout - elapsed,
                    )
                # Probe window elapsed — allow one request through
                self._state   = _State.HALF_OPEN
                self._probing = True
                logger.info("circuit_half_open name=%s", self.name)
            elif self._state == _State.HALF_OPEN:
                # Only one probe at a time; all others fast-fail
                if self._probing:
                    raise CircuitOpenError(self.name, retry_after=self.reset_timeout)

        try:
            result = await fn(*args, **kwargs)
        except Exception as exc:
            async with self._lock:
                self._probing    = False
                self._fail_count += 1
                if self._state == _State.HALF_OPEN or self._fail_count >= self.fail_max:
                    self._state     = _State.OPEN
                    self._opened_at = time.monotonic()
                    logger.error(
                        "circuit_opened name=%s fail_count=%d exc=%s",
                        self.name, self._fail_count, exc,
                    )
            raise

        # Success — reset
        async with self._lock:
            if self._state != _State.CLOSED or self._fail_count > 0:
                logger.info(
                    "circuit_reset name=%s previous_state=%s fail_count=%d",
                    self.name, self._state.value, self._fail_count,
                )
            self._state      = _State.CLOSED
            self._fail_count = 0
            self._opened_at  = None
            self._probing    = False

        return result

    def to_dict(self) -> dict:
        return {
            "name":       self.name,
            "state":      self._state.value,
            "fail_count": self._fail_count,
            "opened_at":  self._opened_at,
        }
