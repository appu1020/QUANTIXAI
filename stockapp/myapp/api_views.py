"""
api_views.py — Production REST API for QuantixAI.

All endpoints return JSON. Authentication is Django session-based.
CSRF is exempt on POST endpoints for API clients; production deployments
should add token authentication.

Endpoints:
  GET  /api/predict/             — Run inference and return prediction
  POST /api/train/               — Trigger model training
  GET  /api/backtest/            — Run backtest simulation
  GET  /api/live-data/           — Latest price quote
  GET  /api/candle-data/         — OHLCV data for charting
  GET  /api/sentiment/           — News sentiment analysis
  POST /api/rag-query/           — AI financial assistant query
  GET  /api/model-performance/   — Model metrics report
  GET  /api/explain-prediction/  — Explainability for latest prediction
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .backtest_engine import run_backtest
from .data_pipeline import (
    download_market_data,
    fetch_candle_data,
    fetch_live_quote,
    fetch_live_quotes_batch,
    invalidate_cache,
)
from .model_engine import MODEL_FILE_MAP, ModelLoader, load_metrics_report
from .prediction_engine import infer
from .sentiment_engine import aggregate_sentiment, fetch_news_for_symbol
from .training_engine import train_models

logger = logging.getLogger(__name__)
MODEL_DIR = Path(settings.BASE_DIR) / "myapp" / "models"


def _json_error(message: str, status: int = 500) -> JsonResponse:
    logger.error("API error [%d]: %s", status, message)
    return JsonResponse({"error": message, "status": "error"}, status=status)


def _get_symbol_interval(request) -> tuple[str, str]:
    symbol = request.GET.get("symbol", "AAPL").upper().strip()
    interval = request.GET.get("interval", "1d").strip()
    return symbol, interval


# ── /api/predict/ ──────────────────────────────────────────────────────────
@require_http_methods(["GET"])
def predict(request):
    """
    Run model inference for a stock symbol.

    Query params:
        symbol   (str): Ticker symbol. Default: AAPL
        interval (str): Data interval. Default: 1d
    """
    symbol, interval = _get_symbol_interval(request)
    try:
        result = infer(symbol=symbol, interval=interval, model_dir=MODEL_DIR)
        resp = JsonResponse({
            "status":          "ok",
            "symbol":          result.symbol,
            "interval":        result.interval,
            "current_price":   round(result.current_price, 4),
            "ensemble_price":  round(result.ensemble_price, 4),
            "predictions":     {k: round(v, 4) for k, v in result.predictions.items()},
            "confidence":      round(result.confidence, 2),
            "signal":          result.direction,
            "probability":     result.probability,
            "horizon_prices":  result.horizon_prices,
            "explanation":     result.explanation,
            "indicator_signals": result.indicator_signals,
        })
        resp["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp["Pragma"] = "no-cache"
        return resp
    except FileNotFoundError as exc:
        return _json_error(str(exc), status=404)
    except Exception as exc:
        logger.exception("Prediction API error")
        return _json_error(str(exc))


# ── /api/train/ ────────────────────────────────────────────────────────────
@require_http_methods(["POST"])
@csrf_exempt
def train(request):
    """
    Trigger model training pipeline.

    Body JSON:
        symbol           (str):  Ticker symbol. Default: AAPL
        interval         (str):  Data interval. Default: 1d
        epochs           (int):  Max training epochs. Default: 20
        batch_size       (int):  Mini-batch size. Default: 32
        train_classifiers(bool): Train directional classifiers too. Default: true
    """
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except json.JSONDecodeError:
        return _json_error("Invalid JSON body", status=400)

    symbol   = payload.get("symbol", "AAPL").upper()
    interval = payload.get("interval", "1d")
    epochs   = int(payload.get("epochs", 20))
    batch_size = int(payload.get("batch_size", 32))
    train_cls  = bool(payload.get("train_classifiers", True))

    try:
        report = train_models(
            symbol=symbol,
            interval=interval,
            epochs=epochs,
            batch_size=batch_size,
            model_dir=MODEL_DIR,
            train_classifiers=train_cls,
        )
        return JsonResponse({
            "status":   "ok",
            "symbol":   symbol,
            "interval": interval,
            "report":   report,
        })
    except Exception as exc:
        logger.exception("Training API error")
        return _json_error(str(exc))


# ── /api/backtest/ ────────────────────────────────────────────────────────
@require_http_methods(["GET"])
def backtest(request):
    """
    Run EMA-crossover backtest for a symbol.

    Query params:
        symbol      (str):   Ticker symbol
        interval    (str):   Data interval
        stop_loss   (float): Stop loss percentage (0.0–1.0). Default: 0.02
        take_profit (float): Take profit percentage (0.0–1.0). Default: 0.04
    """
    symbol, interval = _get_symbol_interval(request)
    stop_loss   = float(request.GET.get("stop_loss", "0.02"))
    take_profit = float(request.GET.get("take_profit", "0.04"))

    try:
        df = download_market_data(symbol=symbol, interval=interval)
        df_reset = df.reset_index()
        if "index" in df_reset.columns:
            df_reset = df_reset.rename(columns={"index": "Date"})
        elif df_reset.columns[0] not in ("Date",):
            df_reset = df_reset.rename(columns={df_reset.columns[0]: "Date"})

        result = run_backtest(
            df=df_reset,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
        )
        return JsonResponse({
            "status":   "ok",
            "symbol":   symbol,
            "interval": interval,
            "backtest": {
                "trades":         result.trades,
                "win_rate":       round(result.win_rate, 4),
                "profit_factor":  round(result.profit_factor, 4),
                "sharpe_ratio":   round(result.sharpe_ratio, 4),
                "max_drawdown":   round(result.max_drawdown, 4),
                "start_date":     str(result.start_date),
                "end_date":       str(result.end_date),
                "summary":        result.summary,
            },
        })
    except Exception as exc:
        logger.exception("Backtest API error")
        return _json_error(str(exc))


# ── /api/live-data/ ───────────────────────────────────────────────────────
@require_http_methods(["GET"])
def live_data(request):
    """
    FIX M2/M5: Return latest price quote using fast_info (sub-300ms).
    Previously used download_market_data() fetching 5y of history just to
    read the last row — extremely wasteful and slow.

    Query params:
        symbol (str): Ticker symbol
    """
    symbol = request.GET.get("symbol", "AAPL").upper().strip()
    try:
        quote = fetch_live_quote(symbol)
        resp = JsonResponse({
            "status":   "ok",
            "symbol":   symbol,
            "latest":   quote,
        })
        # Prevent browser and CDN caching for live data
        resp["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp["Pragma"] = "no-cache"
        resp["Expires"] = "0"
        return resp
    except Exception as exc:
        logger.exception("Live data API error")
        return _json_error(str(exc))


# ── /api/price-stream/ ────────────────────────────────────────────────────
@require_http_methods(["GET"])
def price_stream(request):
    """
    Batch live price endpoint for multiple symbols at once.
    Used by the live ticker strip and dashboard to update all prices
    in a single network request instead of N separate calls.

    Query params:
        symbols (str): Comma-separated ticker symbols. Max 10.
                       e.g. ?symbols=AAPL,MSFT,TSLA,SPY,QQQ
    """
    raw = request.GET.get("symbols", "SPY,QQQ,BTC-USD")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()][:10]
    try:
        quotes = fetch_live_quotes_batch(symbols)
        resp = JsonResponse({
            "status": "ok",
            "quotes": quotes,
            "count":  len(quotes),
        })
        resp["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp["Pragma"] = "no-cache"
        return resp
    except Exception as exc:
        logger.exception("Price stream API error")
        return _json_error(str(exc))


# ── /api/invalidate-cache/ ────────────────────────────────────────────────
@require_http_methods(["POST"])
@csrf_exempt
def invalidate_market_cache(request):
    """
    Clear the server-side market data cache.
    Useful when you want to force-refresh stale data.

    Body JSON:
        symbol   (str): Optional. Clear only this symbol's cache.
        interval (str): Optional. Clear only this interval's cache.
    """
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except json.JSONDecodeError:
        payload = {}
    symbol = payload.get("symbol", "").upper() or None
    interval = payload.get("interval", "") or None
    invalidate_cache(symbol=symbol, interval=interval)
    return JsonResponse({"status": "ok", "message": "Cache invalidated"})


# ── /api/candle-data/ ─────────────────────────────────────────────────────
@require_http_methods(["GET"])
def candle_data(request):
    """
    Return OHLCV candlestick data compatible with TradingView Lightweight Charts.

    Query params:
        symbol   (str): Ticker symbol
        interval (str): Data interval
        limit    (int): Max candles to return. Default: 500
    """
    symbol, interval = _get_symbol_interval(request)
    limit = int(request.GET.get("limit", 500))
    try:
        candles = fetch_candle_data(symbol=symbol, interval=interval)
        candles = candles[-limit:]
        resp = JsonResponse({
            "status":   "ok",
            "symbol":   symbol,
            "interval": interval,
            "count":    len(candles),
            "candles":  candles,
        })
        resp["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp["Pragma"] = "no-cache"
        return resp
    except Exception as exc:
        logger.exception("Candle data API error")
        return _json_error(str(exc))


# ── /api/sentiment/ ───────────────────────────────────────────────────────
@require_http_methods(["GET"])
def sentiment(request):
    """
    Fetch and analyze news sentiment for a symbol.

    Query params:
        symbol  (str): Ticker symbol
        api_key (str): Optional NewsData.io API key override
    """
    symbol = request.GET.get("symbol", "AAPL").upper()
    api_key = request.GET.get("api_key") or getattr(settings, "NEWSDATA_API_KEY", "")

    try:
        articles = fetch_news_for_symbol(symbol=symbol, api_key=api_key or None)
        score = aggregate_sentiment(symbol=symbol, articles=articles)
        return JsonResponse({
            "status":        "ok",
            "symbol":        symbol,
            "sentiment": {
                "bullish_score":  score.bullish_score,
                "bearish_score":  score.bearish_score,
                "impact_score":   score.impact_score,
                "summary":        score.summary,
                "article_count":  score.article_count,
                "bullish_count":  score.bullish_count,
                "bearish_count":  score.bearish_count,
                "neutral_count":  score.neutral_count,
            },
            "articles": [
                {
                    "title":           a.title,
                    "source":          a.source,
                    "url":             a.url,
                    "published_at":    a.published_at,
                    "sentiment":       a.sentiment_label,
                    "compound_score":  a.compound_score,
                }
                for a in score.articles[:10]
            ],
        })
    except Exception as exc:
        logger.exception("Sentiment API error")
        return _json_error(str(exc))


# ── /api/rag-query/ ───────────────────────────────────────────────────────
@require_http_methods(["POST"])
@csrf_exempt
def rag_query(request):
    """
    Query the AI Financial Intelligence RAG system.

    Body JSON:
        question           (str): The user's question
        symbol             (str): Optional ticker symbol for context
        include_prediction (bool): Include latest prediction context. Default: true
    """
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except json.JSONDecodeError:
        return _json_error("Invalid JSON body", status=400)

    question = payload.get("question", "").strip()
    symbol   = payload.get("symbol", "").upper().strip() or None
    include_pred = payload.get("include_prediction", True)

    if not question:
        return _json_error("'question' field is required", status=400)

    try:
        from rag.pipeline import get_pipeline
        from rag.context_builder import build_prediction_context

        pipeline = get_pipeline()

        prediction_context = None
        indicator_signals = None

        if include_pred and symbol:
            try:
                result = infer(symbol=symbol, interval="1d", model_dir=MODEL_DIR, persist=False)
                prediction_context = build_prediction_context(
                    symbol=result.symbol,
                    current_price=result.current_price,
                    predicted_price=result.ensemble_price,
                    direction=result.direction,
                    confidence=result.confidence,
                    probability=result.probability,
                    feature_snapshot=result.feature_snapshot,
                    indicator_signals=result.indicator_signals,
                )
                indicator_signals = result.indicator_signals
            except Exception as exc:
                logger.warning("Could not build prediction context for RAG: %s", exc)

        # FIX R7: Use Django session key for conversation memory
        session_key = request.session.session_key
        if not session_key:
            request.session.create()
            session_key = request.session.session_key

        rag_result = pipeline.query(
            question=question,
            symbol=symbol,
            prediction_context=prediction_context,
            indicator_signals=indicator_signals,
            session_key=session_key,
        )

        return JsonResponse({
            "status":          "ok",
            "question":        question,
            "answer":          rag_result["answer"],
            "intent":          rag_result["intent"],
            "sources":         rag_result["sources"],
            "docs_retrieved":  rag_result["docs_retrieved"],
        })
    except Exception as exc:
        logger.exception("RAG query API error")
        return _json_error(str(exc))


# ── /api/model-performance/ ───────────────────────────────────────────────
@require_http_methods(["GET"])
def model_performance(request):
    """Return the latest model evaluation metrics report."""
    try:
        metrics = load_metrics_report(MODEL_DIR)
        if not metrics:
            return _json_error("No model metrics found. Run training first.", status=404)
        return JsonResponse({"status": "ok", "metrics": metrics})
    except Exception as exc:
        logger.exception("Model performance API error")
        return _json_error(str(exc))


# ── /api/explain-prediction/ ──────────────────────────────────────────────
@require_http_methods(["GET"])
def explain_prediction(request):
    """
    Return detailed explainability data for a symbol's latest prediction.

    Query params:
        symbol   (str): Ticker symbol
        interval (str): Data interval
    """
    symbol, interval = _get_symbol_interval(request)
    try:
        result = infer(symbol=symbol, interval=interval, model_dir=MODEL_DIR, persist=False)

        from .explainability import generate_prediction_explanation
        explanation_data = generate_prediction_explanation(
            feature_snapshot=result.feature_snapshot,
            direction=result.direction,
            predicted_price=result.ensemble_price,
            current_price=result.current_price,
            confidence=result.confidence,
        )

        return JsonResponse({
            "status":            "ok",
            "symbol":            symbol,
            "interval":          interval,
            "prediction":        result.direction,
            "confidence":        round(result.confidence, 2),
            "current_price":     round(result.current_price, 4),
            "ensemble_price":    round(result.ensemble_price, 4),
            "probability":       result.probability,
            "horizon_prices":    result.horizon_prices,
            "explanation":       explanation_data,
        })
    except FileNotFoundError as exc:
        return _json_error(str(exc), status=404)
    except Exception as exc:
        logger.exception("Explain prediction API error")
        return _json_error(str(exc))
