"""Unit tests for shared/rate_limiter.py in-process limiter — no Redis required."""
from __future__ import annotations

import time
import pytest
from unittest.mock import patch

# Force in-process path for all tests by ensuring REDIS_URL is unset.
with patch.dict("os.environ", {"REDIS_URL": ""}):
    import importlib
    import shared.rate_limiter as _rl_module


def _reset_buckets():
    """Clear the in-process token buckets between tests."""
    _rl_module._buckets.clear()
    _rl_module._WARNED_INPROCESS = False


def test_first_request_passes():
    _reset_buckets()
    with patch.object(_rl_module.settings, "REDIS_URL", None), \
         patch.object(_rl_module.settings, "RATE_LIMIT_RPM", 20), \
         patch.object(_rl_module.settings, "RATE_LIMIT_BURST", 5):
        _rl_module.check_rate_limit("user-a")  # should not raise


def test_burst_limit_enforced():
    _reset_buckets()
    with patch.object(_rl_module.settings, "REDIS_URL", None), \
         patch.object(_rl_module.settings, "RATE_LIMIT_RPM", 60), \
         patch.object(_rl_module.settings, "RATE_LIMIT_BURST", 3):
        _rl_module.check_rate_limit("user-b")
        _rl_module.check_rate_limit("user-b")
        _rl_module.check_rate_limit("user-b")
        with pytest.raises(_rl_module.RateLimitExceeded) as exc_info:
            _rl_module.check_rate_limit("user-b")
        assert exc_info.value.retry_after > 0
        assert exc_info.value.user_id == "user-b"


def test_different_users_isolated():
    _reset_buckets()
    with patch.object(_rl_module.settings, "REDIS_URL", None), \
         patch.object(_rl_module.settings, "RATE_LIMIT_RPM", 60), \
         patch.object(_rl_module.settings, "RATE_LIMIT_BURST", 1):
        _rl_module.check_rate_limit("user-x")
        # user-y's bucket is independent
        _rl_module.check_rate_limit("user-y")


def test_tokens_refill_over_time():
    _reset_buckets()
    with patch.object(_rl_module.settings, "REDIS_URL", None), \
         patch.object(_rl_module.settings, "RATE_LIMIT_RPM", 60), \
         patch.object(_rl_module.settings, "RATE_LIMIT_BURST", 1):
        _rl_module.check_rate_limit("user-c")
        with pytest.raises(_rl_module.RateLimitExceeded):
            _rl_module.check_rate_limit("user-c")
        # After ~1 second a new token should have accrued (60 rpm = 1/s)
        time.sleep(1.1)
        _rl_module.check_rate_limit("user-c")   # should pass now


def test_rate_limit_exceeded_attrs():
    exc = _rl_module.RateLimitExceeded(user_id="u1", retry_after=12.345)
    assert exc.user_id    == "u1"
    assert exc.retry_after == 12.3   # rounded to 1 decimal
