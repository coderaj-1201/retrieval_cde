"""
Azure client factories.

Auth strategy (production):
  - OpenAI  : DefaultAzureCredential (managed identity). AZURE_OPENAI_API_KEY only for local dev.
  - Search  : DefaultAzureCredential (managed identity). AZURE_SEARCH_API_KEY only for local dev.
  - Cosmos  : DefaultAzureCredential (managed identity). COSMOS_KEY only for local dev.
  - Foundry : DefaultAzureCredential always.

None of the API keys should be present in production ACA environment variables.
Their absence is what forces the managed-identity path — which is the desired state.
"""
from __future__ import annotations

from functools import lru_cache

from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI

from shared.config import settings


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
    # Guard: only call .get_secret_value() when the key is actually present.
    api_key: str | None = (
        settings.AZURE_OPENAI_API_KEY.get_secret_value()
        if settings.AZURE_OPENAI_API_KEY is not None
        else None
    )
    if api_key:
        import logging
        logging.getLogger(__name__).info("openai_auth=api_key (local dev)")
        return AzureOpenAI(
            azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
            api_key=api_key,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
    import logging
    logging.getLogger(__name__).info("openai_auth=managed_identity")
    token_provider = get_bearer_token_provider(
        _credential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
        azure_ad_token_provider=token_provider,
        api_version=settings.AZURE_OPENAI_API_VERSION,
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
