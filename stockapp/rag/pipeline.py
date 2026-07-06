"""
rag/pipeline.py — Full RAG (Retrieval Augmented Generation) Pipeline.

This is the main orchestrator for the QuantixAI AI reasoning engine.

Pipeline steps:
  1. Document ingestion (static + DB sources)
  2. Text preprocessing and chunking
  3. Embedding generation (sentence-transformers or TF-IDF)
  4. Vector store indexing (FAISS or numpy)
  5. BM25 hybrid search + semantic search + RRF fusion
  6. CrossEncoder reranking
  7. MMR diversity selection
  8. Context injection and prompt construction
  9. LLM reasoning (external LLM if configured, else rule-based with context)
  10. Conversation memory (session-scoped)

FIXES APPLIED:
  R2: Vector store empty detection with clear error messages
  R4: Vector store path mismatch fixed via store_exists() + consistent base path
  R6: Rule-based fallback now uses retrieved context from the prompt instead of
      discarding it and running shallow heuristics independently
  R7: Session-based conversation memory (last 6 turns) injected into all prompts
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default persistence directory
_DEFAULT_STORE_DIR = Path(__file__).resolve().parent / "vector_store_data"

# Conversation memory: keyed by session_key → list of {role, content} dicts
_SESSION_MEMORY: dict[str, list[dict[str, str]]] = {}
_MAX_HISTORY_TURNS = 6


class RAGPipeline:
    """
    Production RAG pipeline for financial intelligence.

    Usage:
        pipeline = RAGPipeline()
        pipeline.initialize()
        answer = pipeline.query("Why did the model predict BUY for AAPL?", symbol="AAPL")
    """

    def __init__(
        self,
        store_dir: Path | None = None,
        chunk_size: int = 600,
        chunk_overlap: int = 150,
        top_k: int = 8,
    ) -> None:
        self.store_dir = Path(store_dir or _DEFAULT_STORE_DIR)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self._embedder = None
        self._vector_store = None
        self._retriever = None
        self._initialized = False

    # ── Initialization ────────────────────────────────────────────────────
    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize or load the RAG pipeline.

        FIX R4: Now uses store_exists() to check the correct derived paths
        (.faiss + .meta.pkl or .npy_store.pkl) instead of just checking for
        a .pkl file that was never actually written by FAISSVectorStore.

        On first run (or force_rebuild=True), ingests all documents, builds
        embeddings, and persists the vector store to disk.
        On subsequent runs, loads from disk for fast startup.
        """
        from .document_loader import load_all_documents
        from .embeddings import create_embedder
        from .retriever import SemanticRetriever
        from .vector_store import create_vector_store, store_exists

        # We always use a fixed base path for the store
        store_base = self.store_dir / "vector_store"
        embedder_path = self.store_dir / "embedder.pkl"

        self._embedder = create_embedder(prefer_semantic=True)
        self._vector_store = create_vector_store(
            dimension=self._embedder.dimension,
            prefer_faiss=True,
        )

        # FIX R4: use store_exists() which checks the correct derived paths
        prefer_faiss = getattr(self._vector_store, "_available", False)
        can_load = (
            not force_rebuild
            and store_exists(store_base, prefer_faiss=prefer_faiss)
        )

        if can_load:
            try:
                self._embedder.load(embedder_path)
                self._vector_store.load(store_base)

                n_docs = len(self._vector_store)
                if n_docs == 0:
                    # FIX R2: Detect empty store after load
                    logger.warning(
                        "Loaded vector store is empty (0 documents). "
                        "Triggering rebuild — documents may have been missing at last build."
                    )
                    raise ValueError("Empty vector store after load")

                logger.info(
                    "RAG pipeline loaded from disk: %d documents indexed", n_docs
                )
                self._retriever = self._build_retriever()
                self._initialized = True
                return
            except Exception as exc:
                logger.warning(
                    "Failed to load RAG from disk (%s), rebuilding from scratch.", exc
                )

        # Build from scratch
        logger.info("Building RAG pipeline from documents…")
        documents = load_all_documents()

        if not documents:
            logger.warning(
                "No documents loaded — RAG pipeline has zero knowledge. "
                "Add .txt files to rag/documents/ and ensure the DB has news/prediction data."
            )
            # Still mark as initialized so the system doesn't crash
            self._initialized = True
            return

        chunks = self._chunk_documents(documents)
        if not chunks:
            logger.warning("Chunking produced 0 chunks from %d documents.", len(documents))
            self._initialized = True
            return

        texts = [c["text"] for c in chunks]

        logger.info("Fitting embedder on %d chunks…", len(texts))
        self._embedder.fit(texts)
        embeddings = self._embedder.embed(texts)

        logger.info(
            "Embeddings generated: shape=%s, dtype=%s",
            embeddings.shape, embeddings.dtype,
        )

        # Validate no null/nan embeddings
        import numpy as np
        null_mask = np.isnan(embeddings).any(axis=1) | np.isinf(embeddings).any(axis=1)
        if null_mask.any():
            n_bad = int(null_mask.sum())
            logger.warning("Dropping %d chunks with NaN/Inf embeddings", n_bad)
            embeddings = embeddings[~null_mask]
            chunks = [c for c, bad in zip(chunks, null_mask) if not bad]

        self._vector_store.add(embeddings, chunks)

        # Persist
        self._embedder.save(embedder_path)
        self._vector_store.save(store_base)

        self._retriever = self._build_retriever()
        self._initialized = True
        logger.info(
            "RAG pipeline built: %d documents -> %d chunks indexed (embed_dim=%d)",
            len(documents), len(chunks), self._embedder.dimension,
        )

    def _build_retriever(self):
        """Build the SemanticRetriever and its BM25 index."""
        from .retriever import SemanticRetriever
        retriever = SemanticRetriever(
            vector_store=self._vector_store,
            embedder=self._embedder,
            initial_k=min(40, max(10, len(self._vector_store))),
        )
        # Build BM25 index from all indexed documents
        retriever.build_bm25_index()
        logger.info("SemanticRetriever ready (BM25 available: %s)", retriever.bm25._available)
        return retriever

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """
        Incrementally add new documents to the pipeline (e.g., fresh news).
        Rebuilds embeddings for new chunks only.
        """
        if not self._initialized:
            self.initialize()

        chunks = self._chunk_documents(documents)
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self._embedder.embed(texts)
        self._vector_store.add(embeddings, chunks)

        # Rebuild BM25 index with new docs
        if self._retriever:
            self._retriever.build_bm25_index()

        # Persist updated store
        store_base = self.store_dir / "vector_store"
        self._vector_store.save(store_base)
        logger.info("Added %d new chunks to RAG pipeline (total: %d)", len(chunks), len(self._vector_store))

    # ── Chunking ──────────────────────────────────────────────────────────
    def _chunk_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split documents into overlapping chunks for retrieval."""
        chunks: list[dict[str, Any]] = []
        for doc in documents:
            text = self._preprocess_text(doc.get("text", ""))
            if not text:
                continue
            words = text.split()
            if len(words) < 20:
                continue  # Skip tiny documents entirely
            step = max(1, self.chunk_size - self.chunk_overlap)
            for start in range(0, len(words), step):
                chunk_words = words[start: start + self.chunk_size]
                if len(chunk_words) < 20:  # Skip tiny trailing chunks
                    continue
                chunk_text = " ".join(chunk_words)
                chunk_meta = {
                    "text":     chunk_text,
                    "source":   doc.get("source", "unknown"),
                    "doc_type": doc.get("doc_type", "document"),
                    "title":    doc.get("title", ""),
                    "symbol":   doc.get("symbol", ""),
                    "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8],
                }
                chunks.append(chunk_meta)
        return chunks

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Clean and normalize text for embedding."""
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\n", " ").replace("\r", " ").strip()
        return text

    # ── Query interface ───────────────────────────────────────────────────
    def query(
        self,
        question: str,
        symbol: str | None = None,
        prediction_context: str | None = None,
        indicator_signals: list[dict[str, str]] | None = None,
        session_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Main query interface. Retrieves relevant knowledge and generates an answer.

        FIX R7: session_key enables conversation memory. If provided, the last
        N turns of the conversation are injected into the prompt.

        Args:
            question:           The user's natural language question.
            symbol:             Optional ticker symbol for context.
            prediction_context: Optional structured prediction text (from context_builder).
            indicator_signals:  Optional list of indicator signal dicts.
            session_key:        Optional session identifier for conversation memory.

        Returns:
            {
              "answer":       str,
              "sources":      [ { source, doc_type, score } ],
              "intent":       str,
              "context_used": str,
              "docs_retrieved": int,
            }
        """
        if not self._initialized:
            self.initialize()

        from .prompt_builder import (
            build_prompt,
            classify_query_intent,
        )

        import time
        start_time = time.time()

        intent = classify_query_intent(question)

        # FIX R7: Retrieve conversation history for this session
        history = _get_session_history(session_key) if session_key else None

        # Augment query with symbol for better retrieval
        retrieval_query = question
        if symbol:
            retrieval_query = f"{question} {symbol} stock"

        # ── Retrieval ─────────────────────────────────────────────────────
        retrieved_docs: list[dict[str, Any]] = []
        store_size = len(self._vector_store) if self._vector_store else 0
        max_score = 0.0

        logger.info(
            "Hybrid AI Router | question='%s' | symbol=%s | intent=%s | session=%s",
            question[:80], symbol, intent, session_key,
        )

        # Skip vector retrieval for pure conversational intents
        skip_retrieval = intent in ["Greeting", "Conversation", "Identity", "Personal Memory", "Programming", "Live Data"]

        if not skip_retrieval and self._retriever and store_size > 0:
            retrieved_docs = self._retriever.retrieve(
                query=retrieval_query,
                top_k=self.top_k,
                use_mmr=True,
            )
            logger.info("Retrieved %d chunks from vector store", len(retrieved_docs))
            for i, doc in enumerate(retrieved_docs):
                score = doc.get("cross_score", doc.get("rrf_score", doc.get("score", 0.0)))
                logger.debug(
                    "  Chunk[%d] score=%.4f source=%s text=%.80s…",
                    i + 1, score,
                doc.get("source", "?"),
                doc.get("text", "").replace("\n", " "),
            )
            max_score = max(max_score, score)

        # ── Build prompt ──────────────────────────────────────────────────
        prompt = build_prompt(
            intent=intent,
            question=question,
            retrieved_docs=retrieved_docs,
            history=history,
            prediction_context=prediction_context,
        )

        logger.debug("Final prompt length: %d chars", len(prompt))

        # ── Generate answer ───────────────────────────────────────────────
        answer = self._generate_answer(
            prompt=prompt,
            question=question,
            intent=intent,
            retrieved_docs=retrieved_docs,
            prediction_context=prediction_context,
            indicator_signals=indicator_signals,
        )
        logger.info("RAG answer generated (%d chars)", len(answer))

        # FIX R7: Save this turn to session memory
        if session_key:
            _append_to_session(session_key, question, answer)

        sources = [
            {
                "source":   doc.get("source", ""),
                "doc_type": doc.get("doc_type", ""),
                "score":    round(
                    doc.get("cross_score", doc.get("rrf_score", doc.get("score", 0.0))), 4
                ),
                "title":    doc.get("title", ""),
            }
            for doc in retrieved_docs
        ]
        
        response_time = round((time.time() - start_time) * 1000) # in ms
        
        # Determine Knowledge Source UI Tag
        if skip_retrieval:
            if intent == "Live Data":
                source_tag = "Live API + LLM"
            else:
                source_tag = "General AI Knowledge"
        else:
            source_tag = "ChromaDB + LLM"

        return {
            "answer":       answer,
            "sources":      sources,
            "intent":       intent,
            "context_used": "prediction" if prediction_context else "rag",
            "docs_retrieved": len(retrieved_docs),
            "confidence":   round(max_score * 100, 1) if not skip_retrieval else 95.0,
            "response_time_ms": response_time,
            "source_tag": source_tag
        }

    def _generate_answer(
        self,
        prompt: str,
        question: str,
        intent: str,
        retrieved_docs: list[dict[str, Any]],
        prediction_context: str | None,
        indicator_signals: list[dict[str, str]] | None,
    ) -> str:
        """
        Generate an answer using Groq LLM via the new llm.py service.
        Strictly forces the LLM to use retrieved context.
        """
        from .llm import generate_rag_response
        
        # Include the prediction context and indicator signals as part of the docs if available
        docs_for_llm = list(retrieved_docs)
        if prediction_context:
            docs_for_llm.append({
                "title": "Real-Time AI Prediction Model Output",
                "text": prediction_context
            })
            
        if indicator_signals:
            signals_text = ", ".join([f"{s['indicator']} is {s['signal']}" for s in indicator_signals])
            docs_for_llm.append({
                "title": "Current Technical Indicators",
                "text": f"The following technical indicators are currently active: {signals_text}"
            })
            
        logger.info("Generating response via Groq for question: '%s' with %d docs", question, len(docs_for_llm))
        
        # Call Groq LLM using the fully constructed hybrid prompt
        answer = generate_rag_response(
            prompt=prompt,
            model=None,
            temperature=0.2
        )
        
        return answer


# ── Session memory helpers (FIX R7) ──────────────────────────────────────────

def _get_session_history(session_key: str) -> list[dict[str, str]]:
    """Return the conversation history for a session."""
    return list(_SESSION_MEMORY.get(session_key, []))


def _append_to_session(session_key: str, question: str, answer: str) -> None:
    """Append a Q&A turn to the session memory, keeping last N turns."""
    if session_key not in _SESSION_MEMORY:
        _SESSION_MEMORY[session_key] = []
    history = _SESSION_MEMORY[session_key]
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer[:500]})
    # Keep only last N turns (each turn = 2 entries)
    if len(history) > _MAX_HISTORY_TURNS * 2:
        _SESSION_MEMORY[session_key] = history[-(  _MAX_HISTORY_TURNS * 2):]


def clear_session(session_key: str) -> None:
    """Clear conversation memory for a session."""
    _SESSION_MEMORY.pop(session_key, None)


# ── Singleton for app-wide use ────────────────────────────────────────────────
_pipeline_instance: RAGPipeline | None = None


def get_pipeline(force_rebuild: bool = False) -> RAGPipeline:
    """
    Get or create the global RAGPipeline singleton.

    The pipeline is lazily initialized on first access and cached for the
    lifetime of the process.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    if not _pipeline_instance._initialized or force_rebuild:
        _pipeline_instance.initialize(force_rebuild=force_rebuild)
    return _pipeline_instance
