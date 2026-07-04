"""
feature_engineering.py — Technical feature engineering pipeline.

Transforms raw OHLCV data into a rich feature matrix for ML model training
and inference. Produces both regression targets (next close price) and
classification targets (directional movement).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicator_engine import (
    adx,
    atr,
    bollinger_bands,
    breakout_signals,
    candle_patterns,
    ema,
    macd,
    obv,
    roc,
    rsi,
    sma,
    stochastic_rsi,
    support_resistance,
    trend_strength,
    vwap,
)

REQUIRED_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]

FEATURE_COLUMNS: list[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA20",
    "SMA50",
    "SMA200",
    "EMA20",
    "EMA50",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",
    "ADX",
    "RSI",
    "STOCH_RSI_K",
    "STOCH_RSI_D",
    "ROC",
    "ATR",
    "BBL",
    "BBM",
    "BBU",
    "VWAP",
    "OBV",
    "SUPPORT",
    "RESISTANCE",
    "BREAKOUT_UP",
    "BREAKOUT_DOWN",
    "TREND_STRENGTH",
    "BULLISH_ENGULFING",
    "BEARISH_ENGULFING",
    "HAMMER",
    "SHOOTING_STAR",
]


def ensure_required_columns(df: pd.DataFrame) -> None:
    """Validate that all OHLCV columns are present."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and return a clean feature DataFrame.

    All NaN values are forward-filled then backward-filled. Raises if any
    NaN values remain after imputation (indicates insufficient data).
    """
    df = df.copy()
    ensure_required_columns(df)
    df = df.astype(float)

    close = df["Close"]

    # ── Trend Indicators ──────────────────────────────────────────────────
    df["SMA20"] = sma(close, window=20)
    df["SMA50"] = sma(close, window=50)
    df["SMA200"] = sma(close, window=200)
    df["EMA20"] = ema(close, span=20)
    df["EMA50"] = ema(close, span=50)

    macd_df = macd(close)
    df = df.join(macd_df)

    df["ADX"] = adx(df["High"], df["Low"], close)

    # ── Momentum Indicators ───────────────────────────────────────────────
    df["RSI"] = np.clip(rsi(close), 0, 100)

    stoch_df = stochastic_rsi(close)
    df = df.join(stoch_df)

    df["ROC"] = roc(close)

    # ── Volatility Indicators ─────────────────────────────────────────────
    df["ATR"] = atr(df["High"], df["Low"], close)

    bb_df = bollinger_bands(close)
    df = df.join(bb_df)

    # ── Volume Indicators ─────────────────────────────────────────────────
    df["VWAP"] = vwap(df["High"], df["Low"], close, df["Volume"])
    df["OBV"] = obv(close, df["Volume"])

    # ── Price Action ──────────────────────────────────────────────────────
    sr_df = support_resistance(close)
    df = df.join(sr_df)

    bo_df = breakout_signals(df)
    df = df.join(bo_df)

    df["TREND_STRENGTH"] = trend_strength(df)

    pattern_df = candle_patterns(df)
    df = df.join(pattern_df)

    # ── Select and clean ──────────────────────────────────────────────────
    df = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    # Pandas 2.x compatible fill (replaces deprecated fillna(method=...))
    df = df.ffill().bfill()

    if df.isna().any().any():
        raise ValueError(
            "Feature engineering produced NaN values after forward/backward fill. "
            "Ensure the input DataFrame has sufficient historical rows (≥ 200 recommended)."
        )

    return df


def add_classification_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Append a binary directional target column to the feature DataFrame.

    TARGET_DIRECTION = 1 if price rises over the next `horizon` candles, else 0.
    The last `horizon` rows will be NaN and should be dropped before training.
    """
    df = df.copy()
    future_close = df["Close"].shift(-horizon)
    df["TARGET_DIRECTION"] = (future_close > df["Close"]).astype(float)
    return df


def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    time_steps: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding-window sequences for time-series models.

    Args:
        features: Scaled feature matrix of shape (T, F).
        target:   Target array of shape (T,).
        time_steps: Look-back window length.

    Returns:
        X of shape (N, time_steps, F) and y of shape (N,).
    """
    if len(features) <= time_steps:
        raise ValueError(
            f"Insufficient data for sequence generation: need > {time_steps} rows, "
            f"got {len(features)}."
        )

    X: list[np.ndarray] = []
    y: list[float] = []
    for i in range(len(features) - time_steps):
        X.append(features[i : i + time_steps])
        y.append(target[i + time_steps])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def create_classification_sequences(
    features: np.ndarray,
    direction_target: np.ndarray,
    time_steps: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sequences for directional (classification) models.

    Returns X of shape (N, time_steps, F) and y of shape (N,) with binary labels.
    """
    return create_sequences(features, direction_target, time_steps)


def prepare_inference_window(
    df: pd.DataFrame,
    scaler,
    time_steps: int = 60,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Prepare the most recent time window for model inference.

    Returns:
        last_window: shaped (1, time_steps, F) — ready for deep learning models.
        feature_df:  The full engineered feature DataFrame (for explainability).
    """
    feature_df = engineer_features(df)
    scaled = scaler.transform(feature_df.values)
    last_window = np.expand_dims(scaled[-time_steps:], axis=0)
    return last_window, feature_df
