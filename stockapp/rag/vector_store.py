"""
rag/vector_store.py — Vector store using ChromaDB.

Stores document embeddings and metadata persistently.
Replaces the legacy FAISS/Numpy implementations with a production-grade ChromaDB.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Production vector store using ChromaDB.
    Supports persistent storage and hybrid search compatibility.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._available = False
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Use provided path or fallback to a default in the current directory
            persist_dir = str(path.parent / "chroma_db") if path else "./chroma_db"
            
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            self._collection = self._client.get_or_create_collection(
                name="quantix_financial_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            self._available = True
            logger.info("ChromaDB initialized at %s", persist_dir)
        except ImportError:
            logger.error("chromadb is not installed. Run `pip install chromadb`.")
            self._client = None
            self._collection = None

    def add(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Add document embeddings and their metadata to ChromaDB."""
        if not self._available or not self._collection:
            logger.error("ChromaVectorStore is not available.")
            return

        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Embeddings/metadata length mismatch: {embeddings.shape[0]} vs {len(metadata)}"
            )

        ids = [str(uuid.uuid4()) for _ in range(len(metadata))]
        documents = [m.get("text", "") for m in metadata]
        
        # Chroma metadata values must be strings, ints, or floats
        clean_metadata = []
        for m in metadata:
            clean_m = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_m[k] = v
                else:
                    clean_m[k] = str(v)
            clean_metadata.append(clean_m)

        # Batch insert to avoid exceeding SQLite limits
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self._collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size].tolist(),
                documents=documents[i:i+batch_size],
                metadatas=clean_metadata[i:i+batch_size]
            )
            
        logger.info("ChromaVectorStore: Added %d documents", len(ids))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-K most similar documents by cosine similarity."""
        if not self._available or not self._collection:
            logger.warning("ChromaVectorStore.search called but store is unavailable.")
            return []
            
        if self._collection.count() == 0:
            logger.warning("ChromaVectorStore is empty.")
            return []

        q = query_vector.astype(np.float32).flatten().tolist()
        k = min(top_k, self._collection.count())
        
        results = self._collection.query(
            query_embeddings=[q],
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )

        formatted_results = []
        if results and results["metadatas"] and results["metadatas"][0]:
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]
            
            for i in range(len(metadatas)):
                result = dict(metadatas[i])
                result["text"] = documents[i]
                # Chroma cosine distance: 0 is identical, 1 is orthogonal, 2 is opposite.
                # Convert distance to similarity score (1 - distance)
                result["score"] = 1.0 - distances[i]
                formatted_results.append(result)

        if formatted_results:
            logger.debug(
                "ChromaVectorStore search: top_k=%d, best_score=%.4f, worst_score=%.4f",
                len(formatted_results), formatted_results[0]["score"], formatted_results[-1]["score"],
            )
        return formatted_results

    def __len__(self) -> int:
        if not self._available or not self._collection:
            return 0
        return self._collection.count()

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Return all documents for BM25 indexing."""
        if not self._available or not self._collection or self._collection.count() == 0:
            return []
        
        results = self._collection.get(include=["metadatas", "documents"])
        docs = []
        if results and results.get("metadatas"):
            for i in range(len(results["metadatas"])):
                doc = dict(results["metadatas"][i])
                doc["text"] = results["documents"][i]
                docs.append(doc)
        return docs

    def save(self, path: Path) -> None:
        """ChromaDB is automatically persistent, so this is a no-op."""
        logger.debug("ChromaVectorStore automatically persists data.")

    def load(self, path: Path) -> "ChromaVectorStore":
        """ChromaDB is automatically persistent, so this is a no-op."""
        logger.debug("ChromaVectorStore automatically loads data from disk.")
        return self


def store_exists(path: Path, prefer_faiss: bool = True) -> bool:
    """Check if the ChromaDB persistent directory exists."""
    persist_dir = path.parent / "chroma_db"
    return persist_dir.exists() and persist_dir.is_dir()


def create_vector_store(dimension: int = 768, prefer_faiss: bool = True) -> ChromaVectorStore:
    """Factory: returns the ChromaDB vector store."""
    return ChromaVectorStore()
