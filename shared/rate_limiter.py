"""
Per-user request rate limiter.

Two implementations are provided — the active one is chosen at startup:

  RedisRateLimiter (production, multi-replica safe)
    Uses a Redis sliding-window counter keyed by user_id + minute bucket.
    Requires REDIS_URL in config. Works correctly across any number of replicas.

  InProcessRateLimiter (local dev / single-replica fallback)
    Token-bucket in process memory with threading.Lock.
    Correct only for single-worker deployments — multi-replica deployments
    can multiply the effective limit by the replica count.

The active limiter is chosen in check_rate_limit():
  - If REDIS_URL is set → RedisRateLimiter
  - Else → InProcessRateLimiter (with a WARNING log on first use)

Usage:
    from shared.rate_limiter import check_rate_limit, RateLimitExceeded
    check_rate_limit(user_id)   # raises RateLimitExceeded if throttled
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

from shared.config import settings

logger = logging.getLogger(__name__)

_WARNED_INPROCESS      = False   # log the single-process warning once
_WARNED_REDIS_FALLBACK = False   # log the Redis→in-process fallback once per process start


class RateLimitExceeded(Exception):
    def __init__(self, user_id: str, retry_after: float) -> None:
        self.user_id     = user_id
        self.retry_after = round(retry_after, 1)
        super().__init__(
            f"Rate limit exceeded for user '{user_id}'. "
            f"Retry after {self.retry_after}s."
        )


# ── In-process token bucket (single-worker only) ───────────────────────────────

@dataclass
class _Bucket:
    tokens:      float
    last_refill: float = field(default_factory=time.monotonic)


_buckets: dict[str, _Bucket] = {}
_lock = threading.Lock()


def _inprocess_check(user_id: str) -> None:
    global _WARNED_INPROCESS
    if not _WARNED_INPROCESS:
        logger.warning(
            "rate_limiter=in_process: effective only for single-worker deployments. "
            "Set REDIS_URL to enable distributed rate limiting for multi-replica ACA."
        )
        _WARNED_INPROCESS = True

    rpm         = settings.RATE_LIMIT_RPM
    burst       = settings.RATE_LIMIT_BURST
    refill_rate = rpm / 60.0

    with _lock:
        now    = time.monotonic()
        bucket = _buckets.get(user_id)
        if bucket is None:
            bucket = _Bucket(tokens=burst, last_refill=now)
            _buckets[user_id] = bucket

        elapsed       = now - bucket.last_refill
        bucket.tokens = min(burst, bucket.tokens + elapsed * refill_rate)
        bucket.last_refill = now

        if bucket.tokens < 1.0:
            retry_after = (1.0 - bucket.tokens) / refill_rate
            raise RateLimitExceeded(user_id=user_id, retry_after=retry_after)

        bucket.tokens -= 1.0


# ── Redis sliding-window counter (multi-replica safe) ─────────────────────────

_redis_client = None
_redis_lock   = threading.Lock()


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            import redis as redis_lib
            from shared.config import settings as s
            _redis_client = redis_lib.from_url(
                s.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            _redis_client.ping()
            logger.info("rate_limiter=redis connected url_preview=%s", s.REDIS_URL[:30])
        except Exception as exc:
            logger.error("rate_limiter_redis_connect_failed: %s — falling back to in-process", exc)
            _redis_client = None
    return _redis_client


def _redis_check(user_id: str) -> None:
    """
    Sliding-window counter via Redis.

    Uses two minute-buckets (current + previous) so the window slides
    smoothly rather than resetting hard at each minute boundary.
    A user gets RATE_LIMIT_RPM requests per rolling 60-second window.
    """
    global _WARNED_REDIS_FALLBACK
    client = _get_redis()
    if client is None:
        # Warn once per process activation — not per request — to avoid log flooding.
        if not _WARNED_REDIS_FALLBACK:
            logger.warning(
                "rate_limiter_redis_unavailable: distributed rate limiting is disabled. "
                "Falling back to in-process limiter — limits are enforced per replica only. "
                "Set REDIS_URL to enable multi-replica-safe rate limiting."
            )
            _WARNED_REDIS_FALLBACK = True
        _inprocess_check(user_id)
        return

    rpm   = settings.RATE_LIMIT_RPM
    now   = time.time()
    minute = int(now // 60)
    elapsed_in_minute = now % 60

    current_key  = f"rl:{user_id}:{minute}"
    previous_key = f"rl:{user_id}:{minute - 1}"

    try:
        pipe = client.pipeline()
        pipe.incr(current_key)
        pipe.expire(current_key, 120)      # 2-minute TTL, covers the sliding window
        pipe.get(previous_key)
        results = pipe.execute()

        current_count  = int(results[0])
        previous_count = int(results[2] or 0)

        # Weighted count: weight previous bucket by how far into the current minute we are
        weight         = 1.0 - (elapsed_in_minute / 60.0)
        weighted_count = current_count + math.floor(previous_count * weight)

        if weighted_count > rpm:
            seconds_until_next = 60 - elapsed_in_minute
            raise RateLimitExceeded(user_id=user_id, retry_after=round(seconds_until_next, 1))

    except RateLimitExceeded:
        raise
    except Exception as exc:
        logger.error("rate_limiter_redis_error user=%s: %s — admitting request", user_id, exc)
        # On Redis error, admit the request rather than blocking all users


# ── Public interface ───────────────────────────────────────────────────────────

def check_rate_limit(user_id: str) -> None:
    """
    Consume one rate-limit token for the given user_id.
    Raises RateLimitExceeded if the user has exceeded their quota.

    Uses Redis if REDIS_URL is configured, otherwise falls back to the
    in-process token bucket (single-worker safe only).
    """
    if hasattr(settings, "REDIS_URL") and settings.REDIS_URL:
        _redis_check(user_id)
    else:
        _inprocess_check(user_id)
