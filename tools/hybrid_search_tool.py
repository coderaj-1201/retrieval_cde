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

logger = logging.getLogger(__name__)


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
    table_raw: str          = ""   # non-empty only for chunk_type == "table"


def _embed(text: str) -> list[float]:
    resp = get_openai_client().embeddings.create(
        input=text,
        model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    )
    return resp.data[0].embedding


def hybrid_search(
    query: str,
    domain: str,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,   # e.g. ["paragraph", "table"] to exclude headings
) -> list[SearchDocument]:
    """
    Single index hybrid search with OData domain filter.
    Optionally filter by chunk_type.
    Returns docs sorted by descending semantic reranker score.
    """
    k = top_k or settings.RETRIEVAL_TOP_K
    client = get_search_client()

    odata_filter = f"domain eq 'ops' and is_deleted eq false"
    if chunk_types:
        type_filter = " or ".join(f"chunk_type eq '{t}'" for t in chunk_types)
        odata_filter += f" and ({type_filter})"

    vector_query = VectorizedQuery(
        vector=_embed(query),
        k_nearest_neighbors=k,
        fields="content_vector",
        exhaustive=False,
    )

    try:
        results = client.search(
            search_text=query,
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
        )

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
            for r in results
        ]
        docs.sort(key=lambda d: d.score, reverse=True)
        logger.debug(
            "hybrid_search domain=%s docs=%d top_score=%.3f",
            domain, len(docs), docs[0].score if docs else 0.0,
        )
        return docs

    except Exception as exc:
        logger.error("Search error domain=%s: %s", domain, exc, exc_info=True)
        return []


def fetch_parent_chunk(parent_id: str) -> SearchDocument | None:
    """
    Fetch a parent chunk by id for full-context retrieval.
    Called by retrieval agent when a child chunk is matched —
    sends the parent's full content to the LLM for better answers.
    """
    if not parent_id:
        return None
    client = get_search_client()
    try:
        r = client.get_document(key=parent_id)
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
