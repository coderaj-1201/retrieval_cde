"""
Azure client factories.

Auth strategy (production):
  - OpenAI  : DefaultAzureCredential (managed identity). AZURE_OPENAI_API_KEY only for local dev.
  - Search  : DefaultAzureCredential (managed identity). AZURE_SEARCH_API_KEY only for local dev.
  - Cosmos  : DefaultAzureCredential (managed identity). COSMOS_KEY only for local dev.
  - Foundry : DefaultAzureCredential always.

Endpoint split:
  - AZURE_OPENAI_ENDPOINT                : *.openai.azure.com  — managed identity (production)
  - AZURE_OPENAI_COGNITIVESERVICES_ENDPOINT: *.cognitiveservices.azure.com — API key (local dev)
  Azure AI Foundry exposes both; the cognitiveservices endpoint is required when
  authenticating with an API key.

None of the API keys should be present in production ACA environment variables.
Their absence is what forces the managed-identity path — which is the desired state.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI

from shared.config import settings

logger = logging.getLogger(__name__)


def _credential() -> DefaultAzureCredential:
    return DefaultAzureCredential()


@lru_cache(maxsize=1)
def get_foundry_client() -> AIProjectClient:
    return AIProjectClient(
        endpoint=str(settings.AZURE_FOUNDRY_PROJECT_ENDPOINT),
        credential=_credential(),
    )


@lru_cache(maxsize=1)
def get_openai_client() -> AzureOpenAI:
    api_key: str | None = (
        settings.AZURE_OPENAI_API_KEY.get_secret_value()
        if settings.AZURE_OPENAI_API_KEY is not None
        else None
    )

    if api_key:
        # API-key auth requires the cognitiveservices.azure.com endpoint.
        # The openai.azure.com endpoint only works with managed identity.
        if settings.AZURE_OPENAI_COGNITIVESERVICES_ENDPOINT is None:
            raise ValueError(
                "AZURE_OPENAI_COGNITIVESERVICES_ENDPOINT must be set when "
                "AZURE_OPENAI_API_KEY is provided. "
                "Format: https://<hub-name>.cognitiveservices.azure.com/"
            )
        endpoint = str(settings.AZURE_OPENAI_COGNITIVESERVICES_ENDPOINT).rstrip("/")
        logger.info("openai_auth=api_key endpoint=%s", endpoint)
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            # max_retries=0: tenacity owns retries; SDK default of 2 would
            # multiply attempts (2 × tenacity 3 = 6 calls).
            max_retries=0,
            timeout=30.0,
        )

    # Production: managed identity via openai.azure.com endpoint.
    endpoint = str(settings.AZURE_OPENAI_ENDPOINT).rstrip("/")
    logger.info("openai_auth=managed_identity endpoint=%s", endpoint)
    token_provider = get_bearer_token_provider(
        _credential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        max_retries=0,
        timeout=30.0,
    )


@lru_cache(maxsize=1)
def get_search_client() -> SearchClient:
    # Guard: only call .get_secret_value() when the key is actually present.
    search_key: str | None = (
        settings.AZURE_SEARCH_API_KEY.get_secret_value()
        if settings.AZURE_SEARCH_API_KEY is not None
        else None
    )
    if search_key:
        import logging
        logging.getLogger(__name__).info("search_auth=api_key (local dev)")
        return SearchClient(
            endpoint=str(settings.AZURE_SEARCH_ENDPOINT),
            index_name=settings.AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(search_key),
        )
    import logging
    logging.getLogger(__name__).info("search_auth=managed_identity")
    return SearchClient(
        endpoint=str(settings.AZURE_SEARCH_ENDPOINT),
        index_name=settings.AZURE_SEARCH_INDEX,
        credential=_credential(),
    )


@lru_cache(maxsize=1)
def get_search_index_client() -> SearchIndexClient:
    search_key: str | None = (
        settings.AZURE_SEARCH_API_KEY.get_secret_value()
        if settings.AZURE_SEARCH_API_KEY is not None
        else None
    )
    if search_key:
        return SearchIndexClient(
            endpoint=str(settings.AZURE_SEARCH_ENDPOINT),
            credential=AzureKeyCredential(search_key),
        )
    return SearchIndexClient(
        endpoint=str(settings.AZURE_SEARCH_ENDPOINT),
        credential=_credential(),
    )
