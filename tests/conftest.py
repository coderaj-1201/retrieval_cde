"""
Shared test configuration.
Sets required environment variables so Pydantic Settings doesn't raise
ValidationError when no .env file is present in the test environment.
"""
import os
import sys
from unittest.mock import MagicMock

# Minimal env vars to satisfy required Pydantic fields before any import
_TEST_ENV = {
    "AZURE_FOUNDRY_PROJECT_ENDPOINT": "https://test-foundry.cognitiveservices.azure.com/",
    "AZURE_OPENAI_ENDPOINT":          "https://test-openai.openai.azure.com/",
    "AZURE_SEARCH_ENDPOINT":          "https://test-search.search.windows.net/",
    "COSMOS_ENDPOINT":                "https://test-cosmos.documents.azure.com:443/",
    "INTERNAL_API_SECRET":            "test-secret-for-unit-tests",
    "REDIS_URL":                      "",
    "ENVIRONMENT":                    "development",
    "MICROSOFT_APP_ID":               "test-app-id",
    "MICROSOFT_APP_PASSWORD":         "test-app-password",
}

for k, v in _TEST_ENV.items():
    os.environ.setdefault(k, v)

# Stub agent_framework before any agent module is imported
_af_stub = MagicMock()
_af_stub.step     = lambda fn: fn
_af_stub.workflow = MagicMock(return_value=lambda fn: fn)
sys.modules.setdefault("agent_framework", _af_stub)
sys.modules.setdefault("retrieval_pipeline.agent_framework", _af_stub)
