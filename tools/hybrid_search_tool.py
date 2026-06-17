"""
Hybrid Search — single index with domain metadata filter.
BM25 + dense vector with RRF fusion + Azure AI Search semantic reranker.

Returns rich metadata from the expanded schema:
  - parent_id     : fetch parent chunk for full context
  - section_heading / section_subheading : shown in source citations
  - page_number   : shown in source citations
  - table_raw     : passed to LLM when chunk_type == table
  - doc_name, doc_url : source attribution
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from azure.search.documents.models import VectorizedQuery

from shared.azure_clients import get_openai_client, get_search_client
from shared.config import settings
from shared.models import Domain
from shared.retry import llm_retry, search_retry

logger = logging.getLogger(__name__)

# Allowlist guards against OData injection — must stay in sync with Domain enum.
_VALID_DOMAINS: frozenset[str] = frozenset(d.value for d in Domain)

# Fields that every search document must carry for synthesis to work correctly.
_REQUIRED_DOC_FIELDS: tuple[str, ...] = ("id", "content")


def _validate_search_doc(r: dict, idx: int) -> bool:
    """Return True if the document has all required fields, else log and return False."""
    for f in _REQUIRED_DOC_FIELDS:
        if not r.get(f):
            logger.warning(
                "search_doc_schema_error idx=%d field_missing=%s doc_id=%s — skipping doc",
                idx, f, r.get("id", "?"),
            )
            return False
    return True


@dataclass(frozen=True)
class SearchDocument:
    # Core (always present)
    id: str
    content: str
    source: str
    domain: str
    score: float
    # Extended (from ingestion schema)
    parent_id: str          = ""
    chunk_type: str         = "paragraph"
    doc_name: str           = ""
    doc_url: str            = ""
    file_type: str          = ""
    page_number: int        = 0
    title: str              = ""
    section_heading: str    = ""
    section_subheading: str = ""
    table_raw: str          = ""


@llm_retry
def _embed(text: str) -> list[float]:
    resp = get_openai_client().embeddings.create(
        input=text,
        model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    )
    return resp.data[0].embedding


@search_retry
def _search(client, search_text: str, vector_query, odata_filter: str, k: int):
    return list(client.search(
        search_text=search_text,
        vector_queries=[vector_query],
        filter=odata_filter,
        query_type="semantic",
        semantic_configuration_name=settings.AZURE_SEARCH_SEMANTIC_CONFIG,
        top=k,
        select=[
            "id", "parent_id", "chunk_type", "domain",
            "doc_name", "source", "doc_url", "file_type",
            "page_number", "title", "section_heading", "section_subheading",
            "content", "table_raw",
        ],
    ))


def hybrid_search(
    query: str,
    domain: str,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
) -> list[SearchDocument]:
    """
    Single index hybrid search with OData domain filter.
    Optionally filter by chunk_type.
    Returns docs sorted by descending semantic reranker score.
    """
    if domain not in _VALID_DOMAINS:
        logger.error(
            "hybrid_search_invalid_domain domain=%r — must be one of %s",
            domain, sorted(_VALID_DOMAINS),
        )
        return []

    k      = top_k or settings.RETRIEVAL_TOP_K
    client = get_search_client()

    # Exclude soft-deleted documents.
    # is_restricted filter is omitted until the field is added to the search index schema.
    odata_filter = f"domain eq '{domain}' and is_deleted eq false"
    if chunk_types:
        type_filter   = " or ".join(f"chunk_type eq '{t}'" for t in chunk_types)
        odata_filter += f" and ({type_filter})"

    try:
        vector_query = VectorizedQuery(
            vector=_embed(query),
            k_nearest_neighbors=k,
            fields="content_vector",
            exhaustive=False,
        )
        raw_results = _search(client, query, vector_query, odata_filter, k)
        docs = [
            SearchDocument(
                id                 = r["id"],
                content            = r.get("content", ""),
                source             = r.get("doc_name") or r.get("source", "unknown"),
                domain             = r.get("domain", domain),
                score              = r.get("@search.reranker_score") or r.get("@search.score", 0.0),
                parent_id          = r.get("parent_id", ""),
                chunk_type         = r.get("chunk_type", "paragraph"),
                doc_name           = r.get("doc_name", ""),
                doc_url            = r.get("doc_url", ""),
                file_type          = r.get("file_type", ""),
                page_number        = r.get("page_number", 0),
                title              = r.get("title", ""),
                section_heading    = r.get("section_heading", ""),
                section_subheading = r.get("section_subheading", ""),
                table_raw          = r.get("table_raw", ""),
            )
            for i, r in enumerate(raw_results)
            if _validate_search_doc(r, i)
        ]
        docs.sort(key=lambda d: d.score, reverse=True)
        logger.debug(
            "hybrid_search domain=%s docs=%d top_score=%.3f",
            domain, len(docs), docs[0].score if docs else 0.0,
        )
        return docs

    except Exception as exc:
        logger.error("hybrid_search_error domain=%s: %s", domain, exc, exc_info=True)
        return []


def fetch_parent_chunk(parent_id: str) -> SearchDocument | None:
    """Fetch a parent chunk by id for full-context retrieval."""
    if not parent_id:
        return None
    client = get_search_client()
    try:
        r = client.get_document(key=parent_id)
        # Respect the is_restricted flag on parent chunks too.
        if r.get("is_restricted") is True:
            logger.warning("fetch_parent_chunk_blocked restricted parent_id=%s", parent_id)
            return None
        return SearchDocument(
            id                 = r["id"],
            content            = r.get("content", ""),
            source             = r.get("doc_name") or r.get("source", "unknown"),
            domain             = r.get("domain", ""),
            score              = 1.0,
            parent_id          = "",
            chunk_type         = r.get("chunk_type", "paragraph"),
            doc_name           = r.get("doc_name", ""),
            doc_url            = r.get("doc_url", ""),
            file_type          = r.get("file_type", ""),
            page_number        = r.get("page_number", 0),
            title              = r.get("title", ""),
            section_heading    = r.get("section_heading", ""),
            section_subheading = r.get("section_subheading", ""),
            table_raw          = r.get("table_raw", ""),
        )
    except Exception:
        return None
