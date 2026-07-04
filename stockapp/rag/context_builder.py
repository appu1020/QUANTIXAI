"""
rag/context_builder.py — Build structured financial context for RAG prompts.

Converts raw prediction results and feature snapshots into human-readable
text that can be matched against the knowledge base via semantic search.
"""

from __future__ import annotations

from typing import Any


def build_prediction_context(
    symbol: str,
    current_price: float,
    predicted_price: float,
    direction: str,
    confidence: float,
    probability: dict[str, float],
    feature_snapshot: dict[str, float],
    indicator_signals: list[dict[str, str]] | None = None,
) -> str:
    """
    Convert a prediction result into a rich text description.

    This text is used to:
      1. Query the vector store for relevant knowledge.
      2. Inject into the prompt as structured context.
    """
    price_change_pct = (predicted_price / max(current_price, 1e-8) - 1.0) * 100

    lines = [
        f"Stock: {symbol}",
        f"Current Price: ${current_price:.2f}",
        f"Predicted Price: ${predicted_price:.2f} ({price_change_pct:+.2f}%)",
        f"Prediction Signal: {direction}",
        f"Model Confidence: {confidence:.1f}%",
        f"BUY Probability: {probability.get('buy', 0):.1f}%",
        f"SELL Probability: {probability.get('sell', 0):.1f}%",
        f"HOLD Probability: {probability.get('hold', 0):.1f}%",
    ]

    # Key indicator values
    if feature_snapshot:
        lines.append("\nKey Technical Indicators:")
        indicator_map = {
            "RSI":           "RSI",
            "MACD":          "MACD",
            "MACD_SIGNAL":   "MACD Signal",
            "ADX":           "ADX",
            "BBL":           "BB Lower",
            "BBU":           "BB Upper",
            "ATR":           "ATR",
            "VWAP":          "VWAP",
            "SMA20":         "SMA20",
            "SMA50":         "SMA50",
            "STOCH_RSI_K":   "Stoch RSI %K",
            "ROC":           "Rate of Change",
            "TREND_STRENGTH": "Trend Strength",
        }
        for key, label in indicator_map.items():
            val = feature_snapshot.get(key)
            if val is not None:
                lines.append(f"  {label}: {val:.4f}")

        # Candlestick patterns present
        patterns = {
            "BULLISH_ENGULFING": "Bullish Engulfing pattern",
            "BEARISH_ENGULFING": "Bearish Engulfing pattern",
            "HAMMER":            "Hammer pattern",
            "SHOOTING_STAR":     "Shooting Star pattern",
            "BREAKOUT_UP":       "Breakout above resistance",
            "BREAKOUT_DOWN":     "Breakout below support",
        }
        active_patterns = [
            label for key, label in patterns.items()
            if feature_snapshot.get(key, 0) == 1
        ]
        if active_patterns:
            lines.append(f"\nActive Patterns: {', '.join(active_patterns)}")

    # Indicator signal summaries
    if indicator_signals:
        lines.append("\nIndicator Signal Summary:")
        for sig in indicator_signals:
            lines.append(f"  [{sig['signal']}] {sig['indicator']}: {sig['explanation'][:120]}")

    return "\n".join(lines)


def build_query_context(query: str, symbol: str | None = None) -> str:
    """
    Augment a user query with symbol context for better retrieval.
    """
    if symbol:
        return f"{query} (context: stock {symbol}, financial market analysis)"
    return f"{query} (financial market analysis context)"
