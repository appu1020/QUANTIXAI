from django.contrib import admin

from .models import (
    BacktestResult,
    HistoricalMarketData,
    NewsArticle,
    PredictionHistory,
    RAGDocument,
    RAGDocumentEmbedding,
    TrainingMetric,
    Wishlist,
)


@admin.register(Wishlist)
class WishlistAdmin(admin.ModelAdmin):
    list_display = ("user", "symbol", "added_on")
    list_filter = ("symbol",)
    search_fields = ("symbol", "user__username")


@admin.register(HistoricalMarketData)
class HistoricalMarketDataAdmin(admin.ModelAdmin):
    list_display = ("symbol", "interval", "timestamp", "close", "volume", "source")
    list_filter = ("symbol", "interval")
    search_fields = ("symbol",)
    ordering = ("symbol", "interval", "timestamp")


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ("symbol", "interval", "generated_at", "signal", "confidence", "current_price")
    list_filter = ("symbol", "interval", "signal")
    ordering = ("-generated_at",)


@admin.register(TrainingMetric)
class TrainingMetricAdmin(admin.ModelAdmin):
    list_display = ("model_name", "model_family", "symbol", "interval", "target", "created_at")
    list_filter = ("model_name", "model_family", "symbol", "interval")
    ordering = ("-created_at",)


@admin.register(BacktestResult)
class BacktestResultAdmin(admin.ModelAdmin):
    list_display = ("symbol", "interval", "strategy_name", "start_date", "end_date", "win_rate", "profit_factor", "sharpe_ratio", "total_return")
    list_filter = ("symbol", "interval", "strategy_name")
    ordering = ("-created_at",)


@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    list_display = ("symbol", "source", "published_at", "sentiment", "bullish_score", "bearish_score", "impact_score")
    list_filter = ("symbol", "source")
    search_fields = ("title", "summary")
    ordering = ("-published_at",)


class RAGDocumentEmbeddingInline(admin.TabularInline):
    model = RAGDocumentEmbedding
    extra = 0
    readonly_fields = ("chunk_index", "embedding_model", "vector_store_id", "created_at")


@admin.register(RAGDocument)
class RAGDocumentAdmin(admin.ModelAdmin):
    list_display = ("source", "document_type", "title", "ingested_at")
    search_fields = ("source", "title")
    inlines = [RAGDocumentEmbeddingInline]
