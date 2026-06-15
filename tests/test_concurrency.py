"""
Adversarial concurrency tests.

Covers: LRU cache race conditions, rate limiter under burst load,
circuit breaker correctness under concurrent failures,
session memory isolation across concurrent conversations.

"ACA runs multiple replicas. Shared state must never corrupt."
"""
from __future__ import annotations

import asyncio
import threading
import time
import sys
from unittest.mock import MagicMock, patch
import pytest

_af_stub = MagicMock()
_af_stub.step     = lambda fn: fn
_af_stub.workflow = MagicMock(return_value=lambda fn: fn)
sys.modules.setdefault("agent_framework", _af_stub)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SESSION LRU CACHE CONCURRENCY
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionLRUCacheConcurrency:

    @pytest.mark.asyncio
    async def test_concurrent_reads_return_correct_session(self):
        """Multiple coroutines reading the same key must all get the same object."""
        from shared.memory import _SessionLRUCache
        from shared.models import SessionMemory

        cache = _SessionLRUCache(max_size=100)
        session = SessionMemory(conversation_id="c-shared", user_id="u-1")
        await cache.set("c-shared", session)

        results = await asyncio.gather(*[cache.get("c-shared") for _ in range(20)])

        assert all(r is session for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_writes_do_not_corrupt_cache(self):
        """50 concurrent coroutines writing different keys must all be retrievable."""
        from shared.memory import _SessionLRUCache
        from shared.models import SessionMemory

        cache = _SessionLRUCache(max_size=200)

        async def _write(i):
            session = SessionMemory(conversation_id=f"c-{i}", user_id=f"u-{i}")
            await cache.set(f"c-{i}", session)
            return f"c-{i}"

        keys = await asyncio.gather(*[_write(i) for i in range(50)])

        # All 50 sessions should be readable
        misses = 0
        for key in keys:
            val = await cache.get(key)
            if val is None:
                misses += 1

        # With max_size=200, all 50 should fit
        assert misses == 0

    @pytest.mark.asyncio
    async def test_lru_eviction_under_concurrent_access(self):
        """When max_size exceeded under concurrent load, cache must not raise."""
        from shared.memory import _SessionLRUCache
        from shared.models import SessionMemory

        cache = _SessionLRUCache(max_size=10)   # tiny cache

        async def _write_and_read(i):
            key = f"c-{i}"
            session = SessionMemory(conversation_id=key, user_id=f"u-{i}")
            await cache.set(key, session)
            # Immediately read back — may have been evicted already
            return await cache.get(key)

        # Write 50 entries to a 10-slot cache concurrently — must not raise
        await asyncio.gather(*[_write_and_read(i) for i in range(50)])

    @pytest.mark.asyncio
    async def test_session_isolation_across_conversations(self):
        """Turns appended for conversation A must not appear in conversation B."""
        from shared.memory import append_turn, load_session, _session_cache
        from shared.models import ConversationTurn, SessionMemory

        async with _session_cache._lock:
            _session_cache._cache.clear()

        session_a = SessionMemory(conversation_id="c-a", user_id="u-1")
        session_b = SessionMemory(conversation_id="c-b", user_id="u-2")

        turn_a = ConversationTurn(
            question_id="qa", answer_id="aa", question="A question", answer="A answer",
            domain="hr", confidence=0.9, tools_used=[],
        )
        turn_b = ConversationTurn(
            question_id="qb", answer_id="ab", question="B question", answer="B answer",
            domain="ops", confidence=0.8, tools_used=[],
        )

        with patch("shared.memory.upsert_document"):
            await asyncio.gather(
                append_turn(session_a, turn_a),
                append_turn(session_b, turn_b),
            )

        assert all(t.question_id == "qa" for t in session_a.turns)
        assert all(t.question_id == "qb" for t in session_b.turns)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  RATE LIMITER UNDER BURST LOAD
# ═══════════════════════════════════════════════════════════════════════════════

class TestRateLimiterConcurrency:

    def test_burst_limit_respected_under_concurrent_threads(self):
        """Burst limit must be enforced even when multiple threads hit simultaneously."""
        import shared.rate_limiter as rl

        rl._buckets.clear()
        rl._WARNED_INPROCESS = True   # suppress startup warning

        passed  = []
        blocked = []
        errors  = []

        def _try_request(user_id="concurrent-user"):
            try:
                with patch.object(rl.settings, "REDIS_URL", None), \
                     patch.object(rl.settings, "RATE_LIMIT_RPM", 60), \
                     patch.object(rl.settings, "RATE_LIMIT_BURST", 5):
                    rl._inprocess_check(user_id)
                    passed.append(1)
            except rl.RateLimitExceeded:
                blocked.append(1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_try_request) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected errors: {errors}"
        # With burst=5, at most 5 should pass immediately
        assert len(passed) <= 5 + 2   # +2 tolerance for timing
        assert len(blocked) >= 13

    def test_different_users_do_not_share_buckets(self):
        """Rate limit for user A must not affect user B."""
        import shared.rate_limiter as rl

        rl._buckets.clear()
        rl._WARNED_INPROCESS = True

        # Exhaust user-A's burst
        with patch.object(rl.settings, "REDIS_URL", None), \
             patch.object(rl.settings, "RATE_LIMIT_RPM", 60), \
             patch.object(rl.settings, "RATE_LIMIT_BURST", 1):
            rl._inprocess_check("user-A")  # consumes the 1 burst token
            with pytest.raises(rl.RateLimitExceeded):
                rl._inprocess_check("user-A")

            # user-B should still have a full bucket
            rl._inprocess_check("user-B")   # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CIRCUIT BREAKER CONCURRENCY
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreakerConcurrency:

    @pytest.mark.asyncio
    async def test_concurrent_failures_open_circuit_exactly_once(self):
        """10 concurrent failures must open the circuit — not cause race conditions."""
        from shared.circuit_breaker import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(name="concurrent-svc", fail_max=3, reset_timeout=60)

        async def _fail():
            raise ValueError("down")

        results = await asyncio.gather(
            *[cb.call(_fail) for _ in range(10)],
            return_exceptions=True,
        )

        value_errors = [r for r in results if isinstance(r, ValueError)]
        circuit_errors = [r for r in results if isinstance(r, CircuitOpenError)]

        # Exactly 3 ValueError (from the closed state), rest CircuitOpenError
        assert len(value_errors)  == 3
        assert len(circuit_errors) == 7
        assert cb.to_dict()["state"] == "open"

    @pytest.mark.asyncio
    async def test_success_after_reset_restores_closed_state(self):
        """After reset timeout, one success must fully close the circuit."""
        from shared.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="recover-svc", fail_max=2, reset_timeout=0.05)

        async def _fail():
            raise ValueError()

        async def _ok():
            return "ok"

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_fail)

        await asyncio.sleep(0.1)   # wait for reset
        await cb.call(_ok)        # HALF_OPEN → success → CLOSED
        assert cb.to_dict()["state"] == "closed"
        assert cb.to_dict()["fail_count"] == 0

    @pytest.mark.asyncio
    async def test_only_first_half_open_probe_is_sent(self):
        """While HALF_OPEN, only 1 probe must be sent — others get CircuitOpenError."""
        from shared.circuit_breaker import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(name="probe-svc", fail_max=2, reset_timeout=0.05)

        async def _fail():
            raise ValueError()

        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_fail)

        await asyncio.sleep(0.1)

        # Transition to HALF_OPEN manually by inspecting the next call
        probe_calls = 0
        open_errors = 0

        async def _slow_ok():
            nonlocal probe_calls
            probe_calls += 1
            await asyncio.sleep(0.05)  # simulate slow probe
            return "ok"

        # Fire 5 concurrent calls while in HALF_OPEN
        results = await asyncio.gather(
            *[cb.call(_slow_ok) for _ in range(5)],
            return_exceptions=True,
        )

        open_errors = sum(1 for r in results if isinstance(r, CircuitOpenError))
        successes   = sum(1 for r in results if r == "ok")

        assert probe_calls == 1         # only one probe sent
        assert successes   == 1
        assert open_errors == 4


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SESSION TURN WINDOW TRIMMING
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionTurnWindow:

    @pytest.mark.asyncio
    async def test_session_trimmed_to_max_turns(self):
        """Session must never exceed SESSION_MAX_TURNS turns — oldest are dropped."""
        from shared.memory import append_turn
        from shared.models import ConversationTurn, SessionMemory

        session = SessionMemory(conversation_id="c-1", user_id="u-1")

        with patch("shared.memory.settings") as cfg, \
             patch("shared.memory.upsert_document"):
            cfg.SESSION_MAX_TURNS = 5

            for i in range(8):
                turn = ConversationTurn(
                    question_id=f"q-{i}", answer_id=f"a-{i}",
                    question=f"Question {i}", answer=f"Answer {i}",
                    domain="ops", confidence=0.9, tools_used=[],
                )
                await append_turn(session, turn)

        assert len(session.turns) == 5
        # Oldest turns (0, 1, 2) must be gone
        ids = [t.question_id for t in session.turns]
        assert "q-0" not in ids
        assert "q-7" in ids   # most recent kept

    @pytest.mark.asyncio
    async def test_session_format_context_shows_last_5_turns(self):
        """format_session_context must include at most 5 turns regardless of session size."""
        from shared.memory import format_session_context
        from shared.models import ConversationTurn, SessionMemory

        session = SessionMemory(conversation_id="c-1", user_id="u-1")
        for i in range(10):
            session.turns.append(ConversationTurn(
                question_id=f"q-{i}", answer_id=f"a-{i}",
                question=f"Q{i}", answer=f"A{i}",
                domain="ops", confidence=0.9, tools_used=[],
            ))

        ctx = format_session_context(session)
        # Only last 5 turns → Q5..Q9
        for i in range(5):
            assert f"Q{i}" not in ctx or i >= 5
        assert "Q9" in ctx
