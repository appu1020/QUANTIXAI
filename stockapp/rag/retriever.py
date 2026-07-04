"""
rag/retriever.py — Semantic retrieval with MMR + BM25 Hybrid + CrossEncoder reranking.

Retrieves relevant documents from the vector store, then applies:
1. BM25 keyword retrieval (if rank_bm25 is installed)
2. Semantic vector search
3. Ensemble fusion (reciprocal rank fusion)
4. CrossEncoder reranking for precision
5. MMR for diversity

FIX R1: CrossEncoder top_k was inverted — max(top_k, len(candidates)) returned
ALL candidates instead of the best top_k. Now correctly limited to top_k.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def mmr_rerank(
    query_vector: np.ndarray,
    candidates: list[dict[str, Any]],
    candidate_embeddings: np.ndarray,
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Maximal Marginal Relevance (MMR) reranking for diversity.

    Balances relevance to query vs. dissimilarity to already-selected documents.
    lambda_param=1.0 → pure relevance; 0.0 → pure diversity.
    Increased default lambda from 0.6 → 0.7 for better relevance.
    """
    if not candidates:
        return []

    selected_indices: list[int] = []
    remaining_indices = list(range(len(candidates)))

    # Relevance scores for all candidates
    query_scores = np.array([
        cosine_similarity(query_vector, candidate_embeddings[i])
        for i in remaining_indices
    ])

    while len(selected_indices) < top_k and remaining_indices:
        if not selected_indices:
            # First pick: highest relevance
            best_i = int(np.argmax([query_scores[i] for i in remaining_indices]))
            chosen_idx = remaining_indices[best_i]
        else:
            # MMR score: λ * relevance - (1-λ) * max_similarity_to_selected
            mmr_scores = []
            for ri in remaining_indices:
                rel_score = cosine_similarity(query_vector, candidate_embeddings[ri])
                sim_to_selected = max(
                    cosine_similarity(candidate_embeddings[ri], candidate_embeddings[si])
                    for si in selected_indices
                )
                mmr_score = lambda_param * rel_score - (1 - lambda_param) * sim_to_selected
                mmr_scores.append(mmr_score)
            chosen_idx = remaining_indices[int(np.argmax(mmr_scores))]

        selected_indices.append(chosen_idx)
        remaining_indices.remove(chosen_idx)

    return [candidates[i] for i in selected_indices]


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank)) across all lists.
    Used to merge semantic search and BM25 keyword search results.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            # Use chunk_id as unique key, fallback to text hash
            doc_id = doc.get("chunk_id") or str(hash(doc.get("text", "")[:100]))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            docs[doc_id] = doc

    # Sort by RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    result = []
    for doc_id in sorted_ids:
        doc = dict(docs[doc_id])
        doc["rrf_score"] = scores[doc_id]
        # Preserve the best individual score for display
        doc["score"] = doc.get("score", scores[doc_id])
        result.append(doc)
    return result


class BM25Retriever:
    """
    BM25 keyword-based retriever. Provides lexical search complementary to semantic search.
    Requires rank_bm25: pip install rank-bm25

    Gracefully degrades to no-op if not installed.
    """

    def __init__(self) -> None:
        self._available = False
        self._bm25 = None
        self._corpus: list[dict[str, Any]] = []

        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
            self._available = True
            logger.info("BM25 keyword retriever available (rank_bm25 installed)")
        except ImportError:
            logger.info(
                "rank_bm25 not installed — BM25 hybrid retrieval disabled. "
                "Run: pip install rank-bm25"
            )

    def index(self, documents: list[dict[str, Any]]) -> None:
        """Build the BM25 index from a list of document dicts with 'text' keys."""
        if not self._available:
            return
        try:
            from rank_bm25 import BM25Okapi
            self._corpus = documents
            tokenized = [doc.get("text", "").lower().split() for doc in documents]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index built: %d documents", len(documents))
        except Exception as exc:
            logger.warning("BM25 indexing failed: %s", exc)
            self._available = False

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Return BM25-ranked documents for the query."""
        if not self._available or self._bm25 is None or not self._corpus:
            return []
        try:
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                result = dict(self._corpus[idx])
                result["score"] = float(scores[idx])
                results.append(result)
            logger.debug("BM25 retrieved %d docs (query: '%s...')", len(results), query[:40])
            return results
        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc)
            return []


class SemanticRetriever:
    """
    Hybrid retriever combining semantic vector search + BM25 keyword search,
    with CrossEncoder reranking and MMR diversity selection.

    Pipeline:
        1. Semantic search (dense vectors)
        2. BM25 keyword search (sparse)
        3. RRF fusion
        4. CrossEncoder reranking (R1 fixed)
        5. MMR diversity selection
    """

    def __init__(self, vector_store, embedder, initial_k: int = 30) -> None:
        """
        Args:
            vector_store: NumpyVectorStore or FAISSVectorStore instance.
            embedder:     TFIDFEmbedder or SentenceEmbedder instance.
            initial_k:    Number of candidates to retrieve before reranking.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.initial_k = initial_k

        from .embeddings import CrossEncoderReranker
        self.cross_encoder = CrossEncoderReranker()
        self.bm25 = BM25Retriever()

    def build_bm25_index(self) -> None:
        """Build BM25 index from all documents in the vector store."""
        if hasattr(self.vector_store, "get_all_documents"):
            docs = self.vector_store.get_all_documents()
            if docs:
                self.bm25.index(docs)
        elif hasattr(self.vector_store, "_metadata") and self.vector_store._metadata:
            self.bm25.index(self.vector_store._metadata)

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        use_mmr: bool = True,
        filter_doc_type: str | None = None,
        similarity_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top_k relevant documents for a query using hybrid search.

        Args:
            query:               Natural language query.
            top_k:               Final number of documents to return.
            use_mmr:             Apply MMR reranking for diversity.
            filter_doc_type:     If set, only return documents of this type.
            similarity_threshold: Minimum score to include a result (0.0 = no filter).

        Returns:
            List of document metadata dicts with 'text' and 'score' keys.
        """
        if len(self.vector_store) == 0:
            logger.warning("Vector store is empty — no documents indexed. "
                           "RAG pipeline needs to be built first.")
            return []

        # ── 1. Semantic search ────────────────────────────────────────────
        query_embedding = self.embedder.embed_query(query)
        semantic_candidates = self.vector_store.search(query_embedding, top_k=self.initial_k)
        logger.debug("Semantic search returned %d candidates", len(semantic_candidates))

        # ── 2. BM25 keyword search ────────────────────────────────────────
        bm25_candidates = self.bm25.search(query, top_k=self.initial_k)
        logger.debug("BM25 search returned %d candidates", len(bm25_candidates))

        # ── 3. Fuse results with RRF ──────────────────────────────────────
        if bm25_candidates:
            # Normalize BM25 scores to [0,1] for comparability
            max_bm25 = max(c.get("score", 1.0) for c in bm25_candidates) or 1.0
            for c in bm25_candidates:
                c["score"] = c.get("score", 0.0) / max_bm25

            fused = reciprocal_rank_fusion([semantic_candidates, bm25_candidates])
        else:
            fused = semantic_candidates

        # ── 4. Apply doc_type filter ──────────────────────────────────────
        if filter_doc_type:
            fused = [c for c in fused if c.get("doc_type") == filter_doc_type]

        # ── 5. Similarity threshold filter ───────────────────────────────
        if similarity_threshold > 0:
            fused = [c for c in fused if c.get("score", 0.0) >= similarity_threshold]

        if not fused:
            logger.warning("No candidates after filtering (threshold=%.2f, doc_type=%s)",
                           similarity_threshold, filter_doc_type)
            return []

        # ── 6. CrossEncoder reranking (FIX R1) ────────────────────────────
        if self.cross_encoder._available:
            # FIX R1: pass top_k correctly — reranker now returns exactly top_k
            fused = self.cross_encoder.rerank(query, fused, top_k=top_k)
            logger.debug("CrossEncoder reranked to top %d", len(fused))
        else:
            fused = fused[:max(top_k * 2, 20)]  # Leave room for MMR

        # ── 7. MMR diversity selection ────────────────────────────────────
        if use_mmr and len(fused) > top_k:
            candidate_texts = [c["text"] for c in fused]
            candidate_embeddings = self.embedder.embed(candidate_texts)
            fused = mmr_rerank(
                query_vector=query_embedding,
                candidates=fused,
                candidate_embeddings=candidate_embeddings,
                top_k=top_k,
            )
        else:
            fused = fused[:top_k]

        logger.info(
            "Retrieved %d documents for query: '%s' "
            "[semantic=%d, bm25=%d, cross_encoder=%s, mmr=%s]",
            len(fused), query[:60],
            len(semantic_candidates), len(bm25_candidates),
            self.cross_encoder._available, use_mmr,
        )

        # Log top scores for debugging
        for i, doc in enumerate(fused[:3]):
            score = doc.get("cross_score", doc.get("rrf_score", doc.get("score", 0.0)))
            logger.debug(
                "  [%d] score=%.4f | source=%s | text=%.60s…",
                i + 1, score, doc.get("source", "?"), doc.get("text", "").replace("\n", " "),
            )

        return fused
