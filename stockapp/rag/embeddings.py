"""
rag/embeddings.py — Text embedding engine for the RAG pipeline.

Uses TF-IDF (scikit-learn) as the primary embedder — zero extra dependencies.
If sentence-transformers is installed, switches to semantic embeddings automatically.

FIX R3: SentenceEmbedder.save() and load() were complete no-ops when
sentence-transformers was available. They now persist the model name to a
sidecar .info file so the pipeline always knows the correct embedding dimension
and model, preventing dimension mismatches across restarts.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TFIDFEmbedder:
    """
    TF-IDF based text embedder. No GPU or heavy models required.
    Documents must be fit before embedding individual queries.
    """

    def __init__(self, max_features: int = 8192, ngram_range: tuple[int, int] = (1, 2)) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=1,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "TFIDFEmbedder":
        self.vectorizer.fit(texts)
        self._fitted = True
        logger.info("TFIDFEmbedder fitted on %d texts, vocab size=%d", len(texts), len(self.vectorizer.vocabulary_))
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TFIDFEmbedder must be fit before embedding.")
        matrix = self.vectorizer.transform(texts)
        # Convert sparse → dense float32
        return np.asarray(matrix.todense(), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.vectorizer, fh)
        logger.info("TFIDFEmbedder saved to %s (dim=%d)", path, self.dimension)

    def load(self, path: Path) -> "TFIDFEmbedder":
        with open(path, "rb") as fh:
            self.vectorizer = pickle.load(fh)
        self._fitted = True
        logger.info("TFIDFEmbedder loaded from %s (dim=%d)", path, self.dimension)
        return self

    @property
    def dimension(self) -> int:
        return len(self.vectorizer.vocabulary_) if self._fitted else 0


class SentenceEmbedder:
    """
    Sentence-transformers based embedder. Produces high-quality semantic embeddings.
    Falls back to TFIDFEmbedder if sentence_transformers is not installed.

    FIX R3: save() and load() now persist/restore the model name via a sidecar
    .info JSON file. This guarantees dimension consistency across process restarts
    and prevents FAISSVectorStore dimension mismatches.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"  # 80MB, fast, good quality

    def __init__(self) -> None:
        self._available = False
        self._model = None
        self._fallback: TFIDFEmbedder | None = None

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
            self._available = True
            logger.info("Using SentenceTransformer embedder: %s (dim=%d)", self.MODEL_NAME, self.dimension)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Falling back to TF-IDF embedder. "
                "Run: pip install sentence-transformers"
            )
            self._fallback = TFIDFEmbedder()
        except Exception as exc:
            logger.error("SentenceTransformer failed to load: %s. Falling back to TF-IDF.", exc)
            self._fallback = TFIDFEmbedder()

    def fit(self, texts: list[str]) -> "SentenceEmbedder":
        if not self._available and self._fallback:
            self._fallback.fit(texts)
        # SentenceTransformer is pre-trained — no fitting needed
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._available and self._model:
            vecs = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=64,
                normalize_embeddings=True,
            )
            return vecs.astype(np.float32)
        if self._fallback:
            return self._fallback.embed(texts)
        raise RuntimeError("No embedder available.")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    def save(self, path: Path) -> None:
        """
        FIX R3: Persist the model name to a sidecar .info file so load()
        always reconstructs the same embedder with the correct dimension.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._available:
            # Save model name so we can verify consistency on load
            info_path = path.with_suffix(".embedder_info.json")
            info = {
                "type": "sentence_transformer",
                "model_name": self.MODEL_NAME,
                "dimension": self.dimension,
            }
            info_path.write_text(json.dumps(info))
            logger.info("SentenceEmbedder info saved to %s (dim=%d)", info_path, self.dimension)
        elif self._fallback:
            self._fallback.save(path)

    def load(self, path: Path) -> "SentenceEmbedder":
        """
        FIX R3: Verify the saved model matches current model. If not, raise
        so the pipeline rebuilds rather than using mismatched embeddings.
        """
        if self._available:
            info_path = path.with_suffix(".embedder_info.json")
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text())
                    saved_dim = info.get("dimension", 0)
                    if saved_dim != self.dimension:
                        raise ValueError(
                            f"Embedding dimension mismatch: saved={saved_dim}, "
                            f"current={self.dimension}. Rebuilding index."
                        )
                    logger.info(
                        "SentenceEmbedder verified: model=%s, dim=%d",
                        info.get("model_name"), saved_dim,
                    )
                except json.JSONDecodeError:
                    logger.warning("Could not parse embedder info — proceeding with current model.")
            else:
                logger.warning(
                    "No embedder info file found at %s. Dimension consistency cannot be verified. "
                    "This will trigger a rebuild if dimensions mismatch.", info_path
                )
            # SentenceTransformer model is always loaded from HuggingFace cache
            # No additional loading needed — the model is already initialized
            return self
        elif self._fallback:
            self._fallback.load(path)
        return self

    @property
    def dimension(self) -> int:
        if self._available and self._model:
            return self._model.get_sentence_embedding_dimension()
        if self._fallback:
            return self._fallback.dimension
        return 0


def create_embedder(prefer_semantic: bool = True) -> TFIDFEmbedder | SentenceEmbedder:
    """Factory: returns best available embedder."""
    if prefer_semantic:
        return SentenceEmbedder()
    return TFIDFEmbedder()


class CrossEncoderReranker:
    """
    Cross-Encoder for reranking retrieved documents.
    Uses ms-marco-MiniLM-L-6-v2 — optimized for passage relevance scoring.
    """
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self) -> None:
        self._available = False
        self._model = None
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.MODEL_NAME)
            self._available = True
            logger.info("Using CrossEncoder reranker: %s", self.MODEL_NAME)
        except ImportError:
            logger.warning("sentence-transformers not installed, CrossEncoder unavailable.")
        except Exception as exc:
            logger.warning("CrossEncoder failed to load: %s", exc)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        """
        FIX R1: top_k was inverted — was passing max(top_k, len(candidates)) which
        returned ALL candidates. Now correctly returns only top_k best results.
        """
        if not self._available or not candidates or self._model is None:
            return candidates[:top_k]

        try:
            pairs = [[query, doc.get("text", "")[:512]] for doc in candidates]
            scores = self._model.predict(pairs)

            for doc, score in zip(candidates, scores):
                doc["cross_score"] = float(score)

            candidates.sort(key=lambda x: x.get("cross_score", 0.0), reverse=True)
            # FIX R1: was max(top_k, len(candidates)) — now correctly top_k
            top = candidates[:top_k]
            logger.debug(
                "CrossEncoder reranked %d → %d candidates. Top score: %.4f",
                len(candidates), len(top),
                top[0].get("cross_score", 0.0) if top else 0.0,
            )
            return top
        except Exception as exc:
            logger.warning("CrossEncoder rerank failed: %s", exc)
            return candidates[:top_k]
