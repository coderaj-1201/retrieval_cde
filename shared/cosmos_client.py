"""
Cosmos DB client factory + container accessors.

Auth priority:
  1. COSMOS_KEY set in env  → key auth  (local dev / simple prod)
  2. COSMOS_KEY not set     → DefaultAzureCredential (managed identity in prod)

The database and containers are NOT auto-created here — run scripts/setup_cosmos.py
once before first deploy. _get_container() will raise clearly if they are missing,
rather than silently swallowing the error.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from azure.cosmos import CosmosClient, ContainerProxy, PartitionKey, exceptions as cosmos_exc
from azure.identity import DefaultAzureCredential

from shared.config import settings

logger = logging.getLogger(__name__)


# ── Client factory ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_cosmos_client() -> CosmosClient:
    key = settings.COSMOS_KEY.get_secret_value()
    if key:
        logger.debug("cosmos_auth=key")
        return CosmosClient(url=str(settings.COSMOS_ENDPOINT), credential=key)
    logger.debug("cosmos_auth=DefaultAzureCredential")
    return CosmosClient(url=str(settings.COSMOS_ENDPOINT), credential=DefaultAzureCredential())


# ── Database accessor — raises clearly if DB missing ──────────────────────────

@lru_cache(maxsize=1)
def _get_database():
    client = get_cosmos_client()
    try:
        db = client.get_database_client(settings.COSMOS_DATABASE)
        # Probe the database so we fail fast on startup rather than on first request
        db.read()
        return db
    except cosmos_exc.CosmosResourceNotFoundError:
        raise RuntimeError(
            f"Cosmos database '{settings.COSMOS_DATABASE}' does not exist. "
            "Run `python scripts/setup_cosmos.py` before deploying."
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Cosmos DB: {exc}") from exc


# ── Container accessor — raises clearly if container missing ──────────────────

def _get_container(container_name: str) -> ContainerProxy:
    db = _get_database()
    try:
        container = db.get_container_client(container_name)
        container.read()   # probe — fails fast if container doesn't exist
        return container
    except cosmos_exc.CosmosResourceNotFoundError:
        raise RuntimeError(
            f"Cosmos container '{container_name}' does not exist in database "
            f"'{settings.COSMOS_DATABASE}'. Run `python scripts/setup_cosmos.py`."
        )


# ── Public accessors (cached per process) ─────────────────────────────────────

@lru_cache(maxsize=1)
def get_chat_container() -> ContainerProxy:
    return _get_container(settings.COSMOS_CONTAINER_CHAT)


@lru_cache(maxsize=1)
def get_feedback_container() -> ContainerProxy:
    return _get_container(settings.COSMOS_CONTAINER_FEEDBACK)


@lru_cache(maxsize=1)
def get_sessions_container() -> ContainerProxy:
    return _get_container(settings.COSMOS_CONTAINER_SESSIONS)


@lru_cache(maxsize=1)
def get_ltm_container() -> ContainerProxy:
    return _get_container(settings.COSMOS_CONTAINER_LTM)


# ── Startup probe — call from lifespan to catch misconfig before first request ─

def probe_cosmos() -> None:
    """
    Call once during app lifespan startup.
    Forces container accessor caching and fails loudly if anything is misconfigured.
    """
    for fn, label in [
        (get_chat_container,     settings.COSMOS_CONTAINER_CHAT),
        (get_feedback_container, settings.COSMOS_CONTAINER_FEEDBACK),
        (get_sessions_container, settings.COSMOS_CONTAINER_SESSIONS),
        (get_ltm_container,      settings.COSMOS_CONTAINER_LTM),
    ]:
        fn()
        logger.info("cosmos_probe_ok container=%s", label)


# ── Generic helpers ────────────────────────────────────────────────────────────

def upsert_document(container: ContainerProxy, doc: dict) -> None:
    """
    Fire-and-forget upsert. Logs on failure, never raises — a Cosmos write
    failure must never take down a query response.
    """
    try:
        container.upsert_item(body=doc)
    except cosmos_exc.CosmosHttpResponseError as exc:
        logger.error(
            "cosmos_upsert_failed container=%s id=%s status=%s: %s",
            container.id, doc.get("id"), exc.status_code, exc.message,
        )
    except Exception as exc:
        logger.error(
            "cosmos_upsert_unexpected container=%s id=%s: %s",
            container.id, doc.get("id"), exc,
        )


def get_document(container: ContainerProxy, item_id: str, partition_key: str) -> dict | None:
    try:
        return container.read_item(item=item_id, partition_key=partition_key)
    except cosmos_exc.CosmosResourceNotFoundError:
        return None
    except cosmos_exc.CosmosHttpResponseError as exc:
        logger.error(
            "cosmos_read_failed container=%s id=%s status=%s: %s",
            container.id, item_id, exc.status_code, exc.message,
        )
        return None
    except Exception as exc:
        logger.error("cosmos_read_unexpected container=%s id=%s: %s", container.id, item_id, exc)
        return None


def query_documents(container: ContainerProxy, query: str, params: list[dict]) -> list[dict]:
    try:
        return list(container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True,
        ))
    except cosmos_exc.CosmosHttpResponseError as exc:
        logger.error(
            "cosmos_query_failed container=%s status=%s query=%.80s: %s",
            container.id, exc.status_code, query, exc.message,
        )
        return []
    except Exception as exc:
        logger.error("cosmos_query_unexpected container=%s query=%.80s: %s", container.id, query, exc)
        return []
