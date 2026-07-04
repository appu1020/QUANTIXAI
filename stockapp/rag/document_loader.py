"""
rag/document_loader.py — Load financial documents for the RAG knowledge base.

Sources:
  1. Static text/JSON files in rag/documents/ (technical analysis, market regimes)
  2. NewsArticle DB records
  3. PredictionHistory DB records (structured as text)

FIX R5: Added JSON document loading, encoding validation, empty-document
detection, and comprehensive logging of what was/wasn't loaded.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"
MIN_DOCUMENT_LENGTH = 50  # Characters — shorter docs are skipped


def load_static_documents() -> list[dict[str, Any]]:
    """
    Load all .txt and .json files from the rag/documents/ directory.

    FIX R5:
    - Validates encoding (UTF-8 with error replace)
    - Skips empty or very short documents (< 50 chars)
    - Logs each loaded document with its character count
    - Supports JSON documents with 'title' and 'content' keys
    """
    docs: list[dict[str, Any]] = []
    if not DOCUMENTS_DIR.exists():
        logger.warning(
            "RAG documents directory not found: %s. "
            "Create this directory and add .txt knowledge files to improve RAG quality.",
            DOCUMENTS_DIR,
        )
        return docs

    # Load .txt files
    for txt_file in sorted(DOCUMENTS_DIR.glob("*.txt")):
        try:
            content = txt_file.read_text(encoding="utf-8", errors="replace")
            content = content.strip()
            if len(content) < MIN_DOCUMENT_LENGTH:
                logger.warning(
                    "Skipping near-empty document: %s (%d chars)", txt_file.name, len(content)
                )
                continue
            docs.append({
                "text":     content,
                "source":   txt_file.name,
                "doc_type": "static_knowledge",
                "title":    txt_file.stem.replace("_", " ").title(),
                "checksum": _checksum(content),
            })
            logger.info("Loaded TXT document: %s (%d chars)", txt_file.name, len(content))
        except Exception as exc:
            logger.error("Failed to load %s: %s", txt_file, exc)

    # Load .json files (expected format: {"title": "...", "content": "..."} or list thereof)
    for json_file in sorted(DOCUMENTS_DIR.glob("*.json")):
        try:
            raw = json_file.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw)
            # Support both single dict and list of dicts
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                content = entry.get("content") or entry.get("text") or ""
                content = str(content).strip()
                title = entry.get("title", json_file.stem.replace("_", " ").title())
                if len(content) < MIN_DOCUMENT_LENGTH:
                    continue
                docs.append({
                    "text":     content,
                    "source":   json_file.name,
                    "doc_type": entry.get("doc_type", "static_knowledge"),
                    "title":    title,
                    "symbol":   entry.get("symbol", ""),
                    "checksum": _checksum(content),
                })
                logger.info("Loaded JSON document: %s | %s (%d chars)", json_file.name, title, len(content))
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in %s: %s", json_file, exc)
        except Exception as exc:
            logger.error("Failed to load %s: %s", json_file, exc)

    logger.info(
        "Loaded %d static knowledge documents from %s (total chars: %d)",
        len(docs), DOCUMENTS_DIR,
        sum(len(d["text"]) for d in docs),
    )
    return docs


def load_news_documents(symbol: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """
    Load recent news articles from the DB as RAG documents.
    Only loads articles with non-empty title + summary (>= 20 chars combined).
    """
    docs: list[dict[str, Any]] = []
    try:
        from myapp.models import NewsArticle
        qs = NewsArticle.objects.all()
        if symbol:
            qs = qs.filter(symbol=symbol)
        qs = qs.order_by("-published_at")[:limit]

        loaded = skipped = 0
        for article in qs:
            text = f"{article.title}. {article.summary or ''}".strip()
            if len(text) < 20:
                skipped += 1
                continue
            docs.append({
                "text":       text[:2000],  # Cap at 2000 chars per article
                "source":     article.source or "news",
                "doc_type":   "news",
                "title":      article.title,
                "symbol":     article.symbol,
                "sentiment":  article.sentiment,
                "published":  str(article.published_at),
                "checksum":   _checksum(text[:200]),
            })
            loaded += 1

        logger.info("Loaded %d news documents from DB (%d skipped as too short)", loaded, skipped)
    except Exception as exc:
        logger.warning("Could not load news documents from DB: %s", exc)
    return docs


def load_prediction_documents(symbol: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Load recent prediction history as structured text for RAG context."""
    docs: list[dict[str, Any]] = []
    try:
        from myapp.models import PredictionHistory
        qs = PredictionHistory.objects.all()
        if symbol:
            qs = qs.filter(symbol=symbol)
        qs = qs.order_by("-generated_at")[:limit]

        for pred in qs:
            try:
                text = (
                    f"Prediction for {pred.symbol} on {pred.generated_at.strftime('%Y-%m-%d %H:%M')}: "
                    f"Signal={pred.signal}, Confidence={pred.confidence:.1f}%, "
                    f"Price=${pred.current_price:.2f}. "
                    f"BUY probability: {pred.buy_probability:.1f}%, "
                    f"SELL probability: {pred.sell_probability:.1f}%, "
                    f"HOLD probability: {pred.hold_probability:.1f}%. "
                    f"Explanation: {pred.explanation}"
                )
                docs.append({
                    "text":     text[:1500],
                    "source":   "prediction_history",
                    "doc_type": "prediction",
                    "symbol":   pred.symbol,
                    "signal":   pred.signal,
                    "checksum": _checksum(text[:200]),
                })
            except Exception:
                continue

        logger.info("Loaded %d prediction documents from DB", len(docs))
    except Exception as exc:
        logger.warning("Could not load prediction documents from DB: %s", exc)
    return docs


def load_all_documents(symbol: str | None = None) -> list[dict[str, Any]]:
    """
    Load all document sources: static knowledge + news + predictions.
    This is the main entry point called during RAG pipeline initialization.
    Deduplicates by checksum to avoid re-indexing identical content.
    """
    all_docs: list[dict[str, Any]] = []
    all_docs.extend(load_static_documents())
    all_docs.extend(load_news_documents(symbol=symbol))
    all_docs.extend(load_prediction_documents(symbol=symbol))

    # Deduplicate by checksum
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for doc in all_docs:
        chk = doc.get("checksum", "")
        if chk and chk in seen:
            continue
        if chk:
            seen.add(chk)
        unique.append(doc)

    deduped = len(all_docs) - len(unique)
    total_chars = sum(len(d["text"]) for d in unique)
    logger.info(
        "Total RAG documents: %d loaded, %d deduplicated -> %d unique | total_chars=%d",
        len(all_docs), deduped, len(unique), total_chars,
    )
    return unique


def _checksum(text: str) -> str:
    """SHA1 short checksum for document deduplication."""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]
