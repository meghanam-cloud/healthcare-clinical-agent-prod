"""
The RAG worker's retrieval tool. Queries the Databricks AI Search index over
the clinical guidelines PDF using native HYBRID mode (ANN + BM25 + RRF done
server-side by Databricks), then reranks the fused candidates locally with a
sentence-transformers cross-encoder - mirroring the pattern in the attached
`end_to_end_rag_databricks.ipynb` reference notebook.
"""

import logging
from functools import lru_cache

from langchain_core.tools import tool
from databricks.ai_search.client import AISearchClient

from src.utils.config_loader import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_index():
    client = AISearchClient()
    return client.get_index(
        endpoint_name=settings.rag.ai_search_endpoint,
        index_name=settings.rag.full_index_name,
    )


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.rag.rerank_model)


def _normalize_results(response) -> list[dict]:
    """Convert the AI Search SDK's tabular response into list[dict]."""
    result = response.get("result", {}) if isinstance(response, dict) else {}
    manifest = response.get("manifest", {}) if isinstance(response, dict) else {}
    data = result.get("data_array", [])
    columns = manifest.get("columns", [])
    if not (columns and data):
        return []
    names = [c["name"] if isinstance(c, dict) else c for c in columns]
    return [dict(zip(names, row)) for row in data]


@tool
def hybrid_search_with_rerank(query: str) -> dict:
    """Retrieve the most relevant clinical guideline chunks for a question.

    Runs Databricks AI Search in HYBRID mode (semantic ANN + BM25 full-text,
    fused server-side with RRF), then reranks the fused candidates with a
    sentence-transformers cross-encoder and keeps the top
    `rag.final_context_k` chunks.

    Args:
        query: The user's natural-language clinical question.

    Returns:
        dict with keys: chunks (list[dict] with text/source/page), error (str | None).
    """
    try:
        index = _get_index()
        response = index.similarity_search(
            query_text=query,
            columns=["id", "text", "source", "page", "chunk_id"],
            num_results=max(settings.rag.ann_k, settings.rag.final_context_k),
            query_type="HYBRID",
        )
        candidates = _normalize_results(response)
        if not candidates:
            return {"chunks": [], "error": None}

        reranker = _get_reranker()
        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = reranker.predict(pairs)

        reranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        top = [c for c, _ in reranked[: settings.rag.final_context_k]]
        return {"chunks": top, "error": None}
    except Exception as e:
        logger.warning("Hybrid RAG retrieval failed: %s", e)
        return {"chunks": [], "error": str(e)}
