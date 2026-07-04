"""
explainability.py — AI Explainability Engine for QuantixAI.

Provides:
  1. Permutation-based feature importance for tree models.
  2. Indicator-based human-readable signal explanations.
  3. Structured explanation dict consumed by the RAG system and frontend.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Indicator interpretation thresholds ───────────────────────────────────
_RSI_THRESHOLDS = {"overbought": 70, "oversold": 30}
_ADX_THRESHOLDS = {"strong_trend": 25, "very_strong": 50}
_MACD_NEUTRAL_BAND = 0.0


def explain_rsi(rsi_value: float) -> dict[str, str]:
    if rsi_value >= _RSI_THRESHOLDS["overbought"]:
        return {
            "indicator": "RSI",
            "value": f"{rsi_value:.1f}",
            "signal": "BEARISH",
            "explanation": (
                f"RSI at {rsi_value:.1f} is in overbought territory (> 70). "
                "This suggests the asset may be overvalued in the short term and due for a pullback."
            ),
        }
    elif rsi_value <= _RSI_THRESHOLDS["oversold"]:
        return {
            "indicator": "RSI",
            "value": f"{rsi_value:.1f}",
            "signal": "BULLISH",
            "explanation": (
                f"RSI at {rsi_value:.1f} is in oversold territory (< 30). "
                "This suggests potential upside as the asset may be undervalued short-term."
            ),
        }
    return {
        "indicator": "RSI",
        "value": f"{rsi_value:.1f}",
        "signal": "NEUTRAL",
        "explanation": f"RSI at {rsi_value:.1f} is in the neutral zone (30–70), indicating balanced buying and selling pressure.",
    }


def explain_macd(macd: float, signal: float, histogram: float) -> dict[str, str]:
    if histogram > 0 and macd > signal:
        return {
            "indicator": "MACD",
            "value": f"MACD={macd:.3f}, Signal={signal:.3f}",
            "signal": "BULLISH",
            "explanation": (
                f"MACD ({macd:.3f}) is above the signal line ({signal:.3f}) with a positive histogram "
                f"({histogram:.3f}). This is a bullish crossover signal indicating upward momentum."
            ),
        }
    elif histogram < 0 and macd < signal:
        return {
            "indicator": "MACD",
            "value": f"MACD={macd:.3f}, Signal={signal:.3f}",
            "signal": "BEARISH",
            "explanation": (
                f"MACD ({macd:.3f}) is below the signal line ({signal:.3f}) with a negative histogram "
                f"({histogram:.3f}). This is a bearish crossover signal indicating downward momentum."
            ),
        }
    return {
        "indicator": "MACD",
        "value": f"MACD={macd:.3f}, Signal={signal:.3f}",
        "signal": "NEUTRAL",
        "explanation": "MACD and signal lines are converging — momentum direction is unclear.",
    }


def explain_bollinger_bands(close: float, bbl: float, bbm: float, bbu: float) -> dict[str, str]:
    bandwidth = (bbu - bbl) / max(bbm, 1e-8)
    if close > bbu:
        return {
            "indicator": "Bollinger Bands",
            "value": f"Close={close:.2f}, Upper={bbu:.2f}",
            "signal": "BEARISH",
            "explanation": (
                f"Price ({close:.2f}) has broken above the upper Bollinger Band ({bbu:.2f}). "
                "This indicates a potential overbought condition; mean reversion may follow."
            ),
        }
    elif close < bbl:
        return {
            "indicator": "Bollinger Bands",
            "value": f"Close={close:.2f}, Lower={bbl:.2f}",
            "signal": "BULLISH",
            "explanation": (
                f"Price ({close:.2f}) has broken below the lower Bollinger Band ({bbl:.2f}). "
                "This indicates a potential oversold condition and possible bounce."
            ),
        }
    pct_position = (close - bbl) / max(bbu - bbl, 1e-8) * 100
    return {
        "indicator": "Bollinger Bands",
        "value": f"Position={pct_position:.0f}% of band",
        "signal": "NEUTRAL",
        "explanation": f"Price is within Bollinger Bands at {pct_position:.0f}% of the range, indicating normal volatility.",
    }


def explain_adx(adx_value: float) -> dict[str, str]:
    if adx_value >= _ADX_THRESHOLDS["very_strong"]:
        label = "very strong"
    elif adx_value >= _ADX_THRESHOLDS["strong_trend"]:
        label = "strong"
    else:
        label = "weak or ranging"
    return {
        "indicator": "ADX",
        "value": f"{adx_value:.1f}",
        "signal": "TRENDING" if adx_value >= _ADX_THRESHOLDS["strong_trend"] else "RANGING",
        "explanation": (
            f"ADX at {adx_value:.1f} indicates a {label} trend. "
            + ("Trend-following strategies are favored." if adx_value >= 25 else "Market may be ranging; trend strategies are less reliable.")
        ),
    }


def explain_volume_obv(obv_current: float, obv_prev: float) -> dict[str, str]:
    change_pct = ((obv_current - obv_prev) / max(abs(obv_prev), 1)) * 100
    if change_pct > 5:
        return {
            "indicator": "OBV (On-Balance Volume)",
            "value": f"OBV Δ={change_pct:+.1f}%",
            "signal": "BULLISH",
            "explanation": f"OBV increased by {change_pct:.1f}%, confirming buying pressure with volume.",
        }
    elif change_pct < -5:
        return {
            "indicator": "OBV (On-Balance Volume)",
            "value": f"OBV Δ={change_pct:+.1f}%",
            "signal": "BEARISH",
            "explanation": f"OBV decreased by {abs(change_pct):.1f}%, indicating selling pressure on volume.",
        }
    return {
        "indicator": "OBV (On-Balance Volume)",
        "value": f"OBV Δ={change_pct:+.1f}%",
        "signal": "NEUTRAL",
        "explanation": "OBV is relatively flat — no strong volume conviction in either direction.",
    }


def explain_support_resistance(close: float, support: float, resistance: float) -> dict[str, str]:
    dist_to_support = (close - support) / max(support, 1e-8) * 100
    dist_to_resistance = (resistance - close) / max(close, 1e-8) * 100
    if dist_to_resistance < 2:
        return {
            "indicator": "Support/Resistance",
            "value": f"Resistance={resistance:.2f} ({dist_to_resistance:.1f}% away)",
            "signal": "BEARISH",
            "explanation": f"Price is within {dist_to_resistance:.1f}% of resistance at {resistance:.2f}. Breakout confirmation needed.",
        }
    elif dist_to_support < 2:
        return {
            "indicator": "Support/Resistance",
            "value": f"Support={support:.2f} ({dist_to_support:.1f}% away)",
            "signal": "BULLISH",
            "explanation": f"Price is near support at {support:.2f} ({dist_to_support:.1f}% away). Potential bounce zone.",
        }
    return {
        "indicator": "Support/Resistance",
        "value": f"S={support:.2f}, R={resistance:.2f}",
        "signal": "NEUTRAL",
        "explanation": f"Price is between support ({support:.2f}) and resistance ({resistance:.2f}) with room to move in either direction.",
    }


# ── Feature importance ────────────────────────────────────────────────────
def compute_permutation_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
) -> list[dict[str, Any]]:
    """
    Compute permutation feature importance for sklearn-compatible models.

    Works with RandomForest, XGBoost, LightGBM. Returns sorted list of
    { feature, importance_mean, importance_std }.
    """
    from sklearn.metrics import mean_squared_error

    baseline_mse = mean_squared_error(y_test, model.predict(X_test))
    importances: dict[str, list[float]] = {name: [] for name in feature_names}

    rng = np.random.default_rng(seed=42)
    for _ in range(n_repeats):
        for col_idx, feat_name in enumerate(feature_names):
            X_permuted = X_test.copy()
            rng.shuffle(X_permuted[:, col_idx])
            permuted_mse = mean_squared_error(y_test, model.predict(X_permuted))
            importances[feat_name].append(permuted_mse - baseline_mse)

    result = [
        {
            "feature": name,
            "importance_mean": float(np.mean(vals)),
            "importance_std":  float(np.std(vals)),
        }
        for name, vals in importances.items()
    ]
    return sorted(result, key=lambda x: x["importance_mean"], reverse=True)


# ── Main explanation generator ────────────────────────────────────────────
def generate_prediction_explanation(
    feature_snapshot: dict[str, float],
    direction: str,
    predicted_price: float,
    current_price: float,
    confidence: float,
) -> dict[str, Any]:
    """
    Generate a full structured explanation for a BUY/SELL/HOLD prediction.

    Args:
        feature_snapshot: Dict of the last row of engineered feature values.
        direction:        'BUY', 'SELL', or 'HOLD'.
        predicted_price:  Ensemble predicted price.
        current_price:    Latest close price.
        confidence:       Ensemble confidence score (0–100).

    Returns:
        {
          "signal": "BUY",
          "summary": "...",
          "price_change_pct": 2.3,
          "indicator_signals": [ { indicator, value, signal, explanation }, ... ],
          "top_factors": [ "RSI overbought", ... ],
          "risk_note": "...",
        }
    """
    price_change_pct = (predicted_price / max(current_price, 1e-8) - 1.0) * 100.0

    indicator_signals: list[dict[str, str]] = []

    rsi_val = feature_snapshot.get("RSI", 50.0)
    indicator_signals.append(explain_rsi(rsi_val))

    macd_val = feature_snapshot.get("MACD", 0.0)
    macd_sig = feature_snapshot.get("MACD_SIGNAL", 0.0)
    macd_hist = feature_snapshot.get("MACD_HIST", 0.0)
    indicator_signals.append(explain_macd(macd_val, macd_sig, macd_hist))

    close = feature_snapshot.get("Close", current_price)
    bbl = feature_snapshot.get("BBL", close * 0.98)
    bbm = feature_snapshot.get("BBM", close)
    bbu = feature_snapshot.get("BBU", close * 1.02)
    indicator_signals.append(explain_bollinger_bands(close, bbl, bbm, bbu))

    adx_val = feature_snapshot.get("ADX", 20.0)
    indicator_signals.append(explain_adx(adx_val))

    support = feature_snapshot.get("SUPPORT", close * 0.95)
    resistance = feature_snapshot.get("RESISTANCE", close * 1.05)
    indicator_signals.append(explain_support_resistance(close, support, resistance))

    # Count signals supporting the prediction direction
    direction_signal_map = {"BUY": "BULLISH", "SELL": "BEARISH"}
    expected_signal = direction_signal_map.get(direction, "NEUTRAL")
    supporting = [s for s in indicator_signals if s["signal"] == expected_signal]
    opposing   = [s for s in indicator_signals if s["signal"] not in (expected_signal, "NEUTRAL", "RANGING", "TRENDING")]

    # Human-readable summary
    top_factors = [s["indicator"] for s in supporting[:3]]
    summary = (
        f"The ensemble model predicts a {direction} signal with {confidence:.1f}% confidence. "
        f"Current price: ${current_price:.2f} → Predicted: ${predicted_price:.2f} "
        f"({price_change_pct:+.2f}%). "
    )
    if top_factors:
        summary += f"Key supporting indicators: {', '.join(top_factors)}."
    if not top_factors:
        summary += "No strong indicator consensus — consider this a low-conviction signal."

    risk_note = _build_risk_note(direction, rsi_val, adx_val, confidence, opposing)

    return {
        "signal":            direction,
        "summary":           summary,
        "price_change_pct":  round(price_change_pct, 4),
        "confidence":        round(confidence, 2),
        "indicator_signals": indicator_signals,
        "top_factors":       top_factors,
        "supporting_count":  len(supporting),
        "opposing_count":    len(opposing),
        "risk_note":         risk_note,
    }


def _build_risk_note(
    direction: str,
    rsi: float,
    adx: float,
    confidence: float,
    opposing_signals: list[dict[str, str]],
) -> str:
    risks = []
    if confidence < 50:
        risks.append("Low model confidence — treat signal as tentative.")
    if direction == "BUY" and rsi > 70:
        risks.append("RSI is overbought — buying into overbought conditions carries elevated reversal risk.")
    if direction == "SELL" and rsi < 30:
        risks.append("RSI is oversold — shorting into oversold conditions may face a bounce.")
    if adx < 20:
        risks.append("ADX below 20 indicates a ranging market — trend signals may be unreliable.")
    if opposing_signals:
        names = [s["indicator"] for s in opposing_signals]
        risks.append(f"Opposing signals from: {', '.join(names)}.")
    if not risks:
        return "No major risk flags identified. Always use stop-loss orders and manage position sizing."
    return " | ".join(risks)
