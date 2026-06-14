"""
In-process token-bucket rate limiter per user_id.

Each user gets a bucket of RATE_LIMIT_BURST tokens that refills at
RATE_LIMIT_RPM tokens per minute. No Redis or external dependency needed —
works correctly for single-worker deployments (uvicorn --workers 1, ACA
with one replica). For multi-replica deployments, promote to Cosmos/Redis.

Usage:
    from shared.rate_limiter import check_rate_limit, RateLimitExceeded
    check_rate_limit(user_id)   # raises RateLimitExceeded if throttled
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from shared.config import settings


class RateLimitExceeded(Exception):
    def __init__(self, user_id: str, retry_after: float) -> None:
        self.user_id    = user_id
        self.retry_after = round(retry_after, 1)
        super().__init__(
            f"Rate limit exceeded for user '{user_id}'. "
            f"Retry after {self.retry_after}s."
        )


@dataclass
class _Bucket:
    tokens:      float
    last_refill: float = field(default_factory=time.monotonic)


_buckets: dict[str, _Bucket] = {}
_lock = threading.Lock()


def check_rate_limit(user_id: str) -> None:
    """
    Consume one token from the user's bucket.
    Raises RateLimitExceeded if the bucket is empty.
    Thread-safe for single-process deployments.
    """
    rpm   = settings.RATE_LIMIT_RPM
    burst = settings.RATE_LIMIT_BURST
    refill_rate = rpm / 60.0   # tokens per second

    now = time.monotonic()

    with _lock:
        bucket = _buckets.get(user_id)
        if bucket is None:
            bucket = _Bucket(tokens=burst)
            _buckets[user_id] = bucket

        # Refill based on elapsed time
        elapsed       = now - bucket.last_refill
        bucket.tokens = min(burst, bucket.tokens + elapsed * refill_rate)
        bucket.last_refill = now

        if bucket.tokens < 1.0:
            # How long until 1 token is available
            retry_after = (1.0 - bucket.tokens) / refill_rate
            raise RateLimitExceeded(user_id=user_id, retry_after=retry_after)

        bucket.tokens -= 1.0
