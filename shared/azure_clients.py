"""
Azure client factories.

Auth:
  - OpenAI: API key (AZURE_OPENAI_API_KEY) if set, else DefaultAzureCredential
  - AI Search: API key via AzureKeyCredential
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


# def _credential() -> DefaultAzureCredential:
#     return DefaultAzureCredential()

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
    # Prefer API key for local dev — avoids RBAC token issues
    api_key = settings.AZURE_OPENAI_API_KEY.get_secret_value() if settings.AZURE_OPENAI_API_KEY else None
    if api_key:
        return AzureOpenAI(
            azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
            api_key=api_key,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
    # Fallback to managed identity (prod)
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
    return SearchClient(
        endpoint=str(settings.AZURE_SEARCH_ENDPOINT),
        index_name=settings.AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY.get_secret_value()),
    )


@lru_cache(maxsize=1)
def get_search_index_client() -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=str(settings.AZURE_SEARCH_ENDPOINT),
        credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY.get_secret_value()),
    )
