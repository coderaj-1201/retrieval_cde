"""
Retry decorators for Azure service calls.

Usage:
    from shared.retry import llm_retry, search_retry

    @llm_retry
    async def my_llm_call(): ...

    @search_retry
    async def my_search_call(): ...
"""
from __future__ import annotations

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Exception types that warrant a retry on OpenAI calls.
# Import lazily so missing package doesn't break the import.
def _openai_transient_exceptions():
    try:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
        return (RateLimitError, APITimeoutError, APIConnectionError)
    except ImportError:
        return (Exception,)


def _search_transient_exceptions():
    try:
        from azure.core.exceptions import HttpResponseError, ServiceRequestError
        return (HttpResponseError, ServiceRequestError)
    except ImportError:
        return (Exception,)


def llm_retry(func):
    """
    Retry an async LLM call up to 3 times with exponential back-off.
    Catches 429 (RateLimitError), timeout, and connection errors.
    Re-raises on the final failure so the caller can handle it.
    """
    return retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_openai_transient_exceptions()),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(func)


def search_retry(func):
    """
    Retry an Azure AI Search call up to 3 times with exponential back-off.
    Catches transient HTTP errors and connection failures.
    Re-raises on the final failure so the caller can return an empty list.
    """
    return retry(
        wait=wait_exponential(multiplier=1, min=1, max=15),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_search_transient_exceptions()),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )(func)
