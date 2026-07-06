"""Compatibility imports for older modules.

Operational finance models now live in ``myapp.models`` so Django migrations,
admin, and app code all share one source of truth.
"""

from .models import (  # noqa: F401
    BacktestResult,
    HistoricalMarketData,
    NewsArticle,
    PredictionHistory,
    RAGDocument,
    RAGDocumentEmbedding,
    TrainingMetric,
)
