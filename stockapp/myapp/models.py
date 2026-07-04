from __future__ import annotations

from django.conf import settings
from django.db import models


class Wishlist(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    symbol = models.CharField(max_length=16, db_index=True)
    added_on = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["user", "symbol"], name="unique_user_symbol_watchlist"),
        ]
        ordering = ["symbol"]

    def __str__(self) -> str:
        return f"{self.user} - {self.symbol}"


class HistoricalMarketData(models.Model):
    symbol = models.CharField(max_length=32, db_index=True)
    interval = models.CharField(max_length=8, db_index=True)
    timestamp = models.DateTimeField(db_index=True)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.FloatField()
    source = models.CharField(max_length=64, default="yfinance")
    validation_flags = models.TextField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["symbol", "interval", "timestamp"], name="unique_market_candle"),
        ]
        indexes = [
            models.Index(fields=["symbol", "interval", "timestamp"]),
        ]
        ordering = ["symbol", "interval", "timestamp"]

    def __str__(self) -> str:
        return f"{self.symbol} {self.interval} {self.timestamp:%Y-%m-%d %H:%M}"


class PredictionHistory(models.Model):
    symbol = models.CharField(max_length=32, db_index=True)
    interval = models.CharField(max_length=8, db_index=True)
    generated_at = models.DateTimeField(auto_now_add=True, db_index=True)
    signal = models.CharField(max_length=16)
    confidence = models.FloatField()
    buy_probability = models.FloatField()
    sell_probability = models.FloatField()
    hold_probability = models.FloatField()
    current_price = models.FloatField()
    horizon_prices = models.TextField(default=dict)
    model_outputs = models.TextField(default=dict)
    feature_snapshot = models.TextField(default=dict)
    explanation = models.TextField(blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "interval", "-generated_at"]),
        ]
        ordering = ["-generated_at"]

    def __str__(self) -> str:
        return f"{self.symbol} {self.signal} {self.confidence:.1f}%"


class TrainingMetric(models.Model):
    model_name = models.CharField(max_length=64, db_index=True)
    model_family = models.CharField(max_length=32, default="regression")
    symbol = models.CharField(max_length=32, db_index=True)
    interval = models.CharField(max_length=8, db_index=True)
    target = models.CharField(max_length=64, default="close")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    metrics = models.TextField(default=dict)
    hyperparameters = models.TextField(default=dict, blank=True)
    artifact_path = models.CharField(max_length=512, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["model_name", "symbol", "interval"]),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.model_name} {self.symbol} {self.interval}"


class BacktestResult(models.Model):
    symbol = models.CharField(max_length=32, db_index=True)
    interval = models.CharField(max_length=8, db_index=True)
    strategy_name = models.CharField(max_length=64)
    start_date = models.DateField()
    end_date = models.DateField()
    trades = models.IntegerField()
    win_rate = models.FloatField()
    profit_factor = models.FloatField()
    sharpe_ratio = models.FloatField()
    max_drawdown = models.FloatField()
    total_return = models.FloatField(default=0.0)
    summary = models.TextField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "interval", "strategy_name"]),
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.strategy_name} {self.symbol} {self.total_return:.2%}"


class NewsArticle(models.Model):
    symbol = models.CharField(max_length=32, blank=True, db_index=True)
    source = models.CharField(max_length=128)
    published_at = models.DateTimeField(null=True, blank=True, db_index=True)
    title = models.CharField(max_length=512)
    summary = models.TextField(blank=True)
    url = models.URLField(blank=True, max_length=1024)
    sentiment = models.FloatField(null=True, blank=True)
    bullish_score = models.FloatField(null=True, blank=True)
    bearish_score = models.FloatField(null=True, blank=True)
    impact_score = models.FloatField(null=True, blank=True)
    raw_payload = models.TextField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["source", "url"], name="unique_news_source_url"),
        ]
        indexes = [
            models.Index(fields=["symbol", "-published_at"]),
        ]
        ordering = ["-published_at", "-created_at"]

    def __str__(self) -> str:
        return self.title[:80]


class RAGDocument(models.Model):
    source = models.CharField(max_length=256, db_index=True)
    document_type = models.CharField(max_length=64, default="document")
    title = models.CharField(max_length=512, blank=True)
    checksum = models.CharField(max_length=64, unique=True)
    metadata = models.TextField(default=dict, blank=True)
    ingested_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-ingested_at"]

    def __str__(self) -> str:
        return self.title or self.source


class RAGDocumentEmbedding(models.Model):
    document = models.ForeignKey(RAGDocument, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.PositiveIntegerField()
    text = models.TextField()
    embedding_model = models.CharField(max_length=128, default="hashing-vectorizer")
    vector_store_id = models.CharField(max_length=128, blank=True)
    metadata = models.TextField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["document", "chunk_index"], name="unique_rag_document_chunk"),
        ]
        ordering = ["document_id", "chunk_index"]

    def __str__(self) -> str:
        return f"{self.document_id}:{self.chunk_index}"
