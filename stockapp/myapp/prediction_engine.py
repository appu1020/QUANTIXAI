"""
prediction_engine.py — Multi-horizon prediction and signal generation.

Combines regression (price level) and classification (direction) models
to produce BUY/SELL/HOLD signals with confidence and probability scores.
Also generates multi-horizon price targets (5m, 15m, 1h).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data_pipeline import download_market_data
from .feature_engineering import prepare_inference_window
from .model_engine import (
    MODEL_FILE_MAP,
    ModelLoader,
    calculate_confidence,
    ensemble_predict,
    load_saved_models,
)

logger = logging.getLogger(__name__)

HORIZON_MULTIPLIERS: dict[str, float] = {
    "5m":  1.005,   # modest projected move
    "15m": 1.010,
    "1h":  1.018,
}


@dataclass
class PredictionResult:
    symbol: str
    interval: str
    current_price: float
    ensemble_price: float
    predictions: dict[str, float]
    confidence: float
    direction: str                          # BUY | SELL | HOLD
    probability: dict[str, float]           # {"buy": 82.3, "sell": 5.1, "hold": 12.6}
    horizon_prices: dict[str, float] = field(default_factory=dict)   # {"5m": x, "15m": x, "1h": x}
    feature_snapshot: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    indicator_signals: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":          self.symbol,
            "interval":        self.interval,
            "current_price":   self.current_price,
            "ensemble_price":  self.ensemble_price,
            "predictions":     self.predictions,
            "confidence":      self.confidence,
            "direction":       self.direction,
            "probability":     self.probability,
            "horizon_prices":  self.horizon_prices,
            "feature_snapshot": self.feature_snapshot,
            "explanation":     self.explanation,
            "indicator_signals": self.indicator_signals,
        }


def load_prediction_artifacts(model_dir: Path) -> tuple[dict[str, Any], Any]:
    loader = ModelLoader(model_dir=model_dir)
    models = loader.get_models()
    scaler = loader.get_scaler()
    if not models:
        raise FileNotFoundError(
            f"No trained model artifacts found in {model_dir}. "
            "Run the training pipeline first: POST /api/train/"
        )
    return models, scaler


def compute_signal(
    predicted_price: float,
    current_price: float,
    confidence: float,
    feature_snapshot: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute BUY/SELL/HOLD probabilities combining price movement and indicator confluence.

    The raw score from price direction is blended with indicator-based scores
    for a more robust signal.
    """
    move_pct = (predicted_price / max(current_price, 1e-8)) - 1.0

    # Base scores from price direction (scaled to 0-100)
    base_buy  = float(np.clip(move_pct * 500 + 50, 0, 100))
    base_sell = float(np.clip(-move_pct * 500 + 50, 0, 100))

    # Indicator confluence adjustments
    indicator_bias = 0.0
    if feature_snapshot:
        rsi = feature_snapshot.get("RSI", 50.0)
        macd_hist = feature_snapshot.get("MACD_HIST", 0.0)
        trend_strength = feature_snapshot.get("TREND_STRENGTH", 0.0)
        breakout_up = feature_snapshot.get("BREAKOUT_UP", 0)
        breakout_down = feature_snapshot.get("BREAKOUT_DOWN", 0)

        if rsi < 35:
            indicator_bias += 10   # oversold → bullish bias
        elif rsi > 65:
            indicator_bias -= 10   # overbought → bearish bias

        if macd_hist > 0:
            indicator_bias += 5
        elif macd_hist < 0:
            indicator_bias -= 5

        if trend_strength > 0:
            indicator_bias += 3
        elif trend_strength < 0:
            indicator_bias -= 3

        if breakout_up:
            indicator_bias += 8
        if breakout_down:
            indicator_bias -= 8

    buy_score  = float(np.clip(base_buy  + indicator_bias, 0, 100))
    sell_score = float(np.clip(base_sell - indicator_bias, 0, 100))

    # Confidence factor: low confidence pushes toward HOLD
    confidence_factor = confidence / 100.0
    hold_score = float(np.clip((100 - max(buy_score, sell_score)) + (1 - confidence_factor) * 20, 0, 100))

    # Normalize to sum to 100
    total = buy_score + sell_score + hold_score
    if total > 0:
        buy_score  = buy_score  / total * 100
        sell_score = sell_score / total * 100
        hold_score = hold_score / total * 100

    return {"buy": round(buy_score, 2), "sell": round(sell_score, 2), "hold": round(hold_score, 2)}


def compute_horizon_prices(
    ensemble_price: float,
    current_price: float,
    interval: str,
) -> dict[str, float]:
    """
    Estimate multi-horizon price targets based on ensemble prediction.

    For intraday intervals, computes 5m/15m/1h targets. For daily, computes
    1d/1w targets. Uses linear interpolation of the predicted price delta.
    """
    delta = ensemble_price - current_price

    if interval in ("1d", "1h"):
        return {
            "1h":  round(current_price + delta * 0.3, 4),
            "4h":  round(current_price + delta * 0.6, 4),
            "1d":  round(ensemble_price, 4),
        }
    else:
        return {
            "5m":  round(current_price + delta * 0.2, 4),
            "15m": round(current_price + delta * 0.5, 4),
            "1h":  round(ensemble_price, 4),
        }


def infer(
    symbol: str,
    interval: str = "1d",
    model_dir: Path | None = None,
    persist: bool = True,
) -> PredictionResult:
    """
    Run the full inference pipeline for a symbol.

    1. Download and engineer features.
    2. Run all available models.
    3. Ensemble predictions.
    4. Generate BUY/SELL/HOLD signal with probabilities.
    5. Compute multi-horizon price targets.
    6. Generate human-readable explanation.
    7. Optionally persist to DB.

    Args:
        symbol:    Ticker symbol.
        interval:  Data interval.
        model_dir: Path to model artifacts directory.
        persist:   Whether to save result to PredictionHistory DB.

    Returns:
        PredictionResult with all computed fields.
    """
    model_dir = Path(model_dir or Path(__file__).resolve().parent / "models")
    models, scaler = load_prediction_artifacts(model_dir)

    df = download_market_data(symbol, interval)
    last_window, feature_df = prepare_inference_window(df, scaler, time_steps=60)
    current_price = float(df["Close"].iloc[-1])

    # Capture feature snapshot (last row)
    feature_snapshot: dict[str, float] = {
        col: float(feature_df[col].iloc[-1])
        for col in feature_df.columns
    }

    # ── Run all models ─────────────────────────────────────────────────────
    predictions: dict[str, float] = {}
    for key, model in models.items():
        # Skip classifier models for regression ensemble
        if key.endswith("_cls"):
            continue
        try:
            flat_input = last_window.reshape(1, -1)
            if key in {"xgboost", "lightgbm", "rf"}:
                pred = float(model.predict(flat_input).reshape(-1)[0])
            else:
                pred = float(model.predict(last_window, verbose=0).reshape(-1)[0])
            predictions[key.upper()] = pred
        except Exception as exc:
            logger.warning("Inference failed for model '%s': %s", key, exc)

    if not predictions:
        raise RuntimeError(
            "Prediction engine could not generate any model outputs. "
            "Ensure models are trained and artifacts exist."
        )

    # ── Inverse-transform ONLY deep-learning model predictions ───────────
    # FIX V2+V3: After training fix (V3), DL models (LSTM/GRU/BiLSTM/CNN-LSTM)
    # are trained on scaled [0,1] targets, so their predictions MUST be
    # inverse-transformed back to real prices.
    # Tree models (rf, xgboost, lightgbm) are trained on raw prices — no transform.
    try:
        from .feature_engineering import FEATURE_COLUMNS
        close_idx = FEATURE_COLUMNS.index("Close")
        price_min = float(scaler.data_min_[close_idx])
        price_max = float(scaler.data_max_[close_idx])
        price_range = max(price_max - price_min, 1e-8)

        DL_KEYS = {"LSTM", "BILSTM", "GRU", "CNN_LSTM"}  # uppercase as stored in predictions
        for k in list(predictions.keys()):
            if k.upper().replace("-", "_") in DL_KEYS:
                raw_val = predictions[k]
                real_val = price_min + raw_val * price_range
                predictions[k] = round(float(real_val), 4)
                logger.info(
                    "Inverse-transformed %s: %.6f → $%.2f (range=[%.2f, %.2f])",
                    k, raw_val, real_val, price_min, price_max,
                )
    except Exception as exc:
        logger.warning("Inverse transform failed, using raw predictions: %s", exc)

    # ── Sanity-filter predictions: discard those too far from current price ─
    # DL models (LSTM/GRU/BiLSTM/CNN-LSTM) may have learned a different price
    # scale during training. Keep only predictions within 60% of current price.
    sane_predictions: dict[str, float] = {}
    for k, v in predictions.items():
        ratio = abs(v - current_price) / max(current_price, 1e-8)
        if ratio <= 0.60:
            sane_predictions[k] = v
        else:
            logger.warning("Dropping outlier prediction %s=%.4f (current=%.2f, ratio=%.2f)",
                           k, v, current_price, ratio)

    if sane_predictions:
        predictions = sane_predictions
    else:
        # All predictions are outliers — fall back to current price as neutral
        logger.warning("All model predictions are outliers; using current price as ensemble fallback")
        predictions = {"FALLBACK": current_price}

    ensemble_price = ensemble_predict(predictions)
    confidence = calculate_confidence(predictions)

    # ── Signal computation ────────────────────────────────────────────────
    signals = compute_signal(ensemble_price, current_price, confidence, feature_snapshot)
    max_signal = max(signals, key=lambda k: signals[k])
    direction = {"buy": "BUY", "sell": "SELL", "hold": "HOLD"}[max_signal]

    # ── Multi-horizon prices ───────────────────────────────────────────────
    horizon_prices = compute_horizon_prices(ensemble_price, current_price, interval)

    # ── Explainability ────────────────────────────────────────────────────
    from .explainability import generate_prediction_explanation
    explanation_data = generate_prediction_explanation(
        feature_snapshot=feature_snapshot,
        direction=direction,
        predicted_price=ensemble_price,
        current_price=current_price,
        confidence=confidence,
    )
    explanation = explanation_data["summary"]
    indicator_signals = explanation_data.get("indicator_signals", [])

    result = PredictionResult(
        symbol=symbol,
        interval=interval,
        current_price=current_price,
        ensemble_price=ensemble_price,
        predictions=predictions,
        confidence=confidence,
        direction=direction,
        probability=signals,
        horizon_prices=horizon_prices,
        feature_snapshot=feature_snapshot,
        explanation=explanation,
        indicator_signals=indicator_signals,
    )

    if persist:
        _persist_prediction(result, explanation_data)

    return result


def _persist_prediction(result: PredictionResult, explanation_data: dict[str, Any]) -> None:
    """Save prediction result to PredictionHistory DB model (best-effort)."""
    try:
        from .models import PredictionHistory
        PredictionHistory.objects.create(
            symbol=result.symbol,
            interval=result.interval,
            signal=result.direction,
            confidence=result.confidence,
            buy_probability=result.probability.get("buy", 0),
            sell_probability=result.probability.get("sell", 0),
            hold_probability=result.probability.get("hold", 0),
            current_price=result.current_price,
            horizon_prices=result.horizon_prices,
            model_outputs=result.predictions,
            feature_snapshot={k: v for k, v in list(result.feature_snapshot.items())[:20]},
            explanation=result.explanation,
        )
    except Exception as exc:
        logger.warning("Could not persist prediction to DB: %s", exc)
