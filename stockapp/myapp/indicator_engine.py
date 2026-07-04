from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs)).fillna(0)


def stochastic_rsi(series: pd.Series, length: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    rsi_series = rsi(series, length=length)
    lowest = rsi_series.rolling(length, min_periods=1).min()
    highest = rsi_series.rolling(length, min_periods=1).max()
    k = 100 * ((rsi_series - lowest) / (highest - lowest).replace(0, np.nan))
    k = k.rolling(smooth_k, min_periods=1).mean()
    d = k.rolling(smooth_d, min_periods=1).mean()
    return pd.DataFrame({"STOCH_RSI_K": k.fillna(0), "STOCH_RSI_D": d.fillna(0)})


def roc(series: pd.Series, length: int = 12) -> pd.Series:
    return 100 * (series / series.shift(length) - 1)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(series, span=fast)
    slow_ema = ema(series, span=slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, span=signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "MACD_SIGNAL": signal_line, "MACD_HIST": histogram})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    return true_range(high, low, close).rolling(length, min_periods=1).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = low.diff().abs()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = true_range(high, low, close)
    atr_series = tr.rolling(length, min_periods=1).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / length, min_periods=1, adjust=False).mean() / atr_series.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, min_periods=1, adjust=False).mean() / atr_series.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, min_periods=1, adjust=False).mean().fillna(0)


def bollinger_bands(series: pd.Series, length: int = 20, std: int = 2) -> pd.DataFrame:
    middle = series.rolling(length, min_periods=1).mean()
    deviation = series.rolling(length, min_periods=1).std()
    return pd.DataFrame({"BBL": middle - std * deviation, "BBM": middle, "BBU": middle + std * deviation})


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum().fillna(0)


def support_resistance(close: pd.Series, window: int = 20) -> pd.DataFrame:
    rolling_low = close.rolling(window, min_periods=1).min()
    rolling_high = close.rolling(window, min_periods=1).max()
    return pd.DataFrame({"SUPPORT": rolling_low, "RESISTANCE": rolling_high})


def candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    body = df["Close"] - df["Open"]
    range_ = df["High"] - df["Low"]
    upper_shadow = df["High"] - df[["Close", "Open"]].max(axis=1)
    lower_shadow = df[["Close", "Open"]].min(axis=1) - df["Low"]

    bullish_engulfing = (
        (body > 0)
        & (body.shift(1) < 0)
        & (df["Open"] < df["Close"].shift(1))
        & (df["Close"] > df["Open"].shift(1))
    )
    bearish_engulfing = (
        (body < 0)
        & (body.shift(1) > 0)
        & (df["Open"] > df["Close"].shift(1))
        & (df["Close"] < df["Open"].shift(1))
    )
    hammer = (body > 0) & (lower_shadow > 2 * body) & (upper_shadow < body)
    shooting_star = (body < 0) & (upper_shadow > 2 * body.abs()) & (lower_shadow < body.abs())
    return pd.DataFrame(
        {
            "BULLISH_ENGULFING": bullish_engulfing.astype(int),
            "BEARISH_ENGULFING": bearish_engulfing.astype(int),
            "HAMMER": hammer.astype(int),
            "SHOOTING_STAR": shooting_star.astype(int),
        }
    )


def breakout_signals(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    support_resistance_df = support_resistance(df["Close"], window=window)
    breakout_up = df["Close"] > support_resistance_df["RESISTANCE"].shift(1)
    breakout_down = df["Close"] < support_resistance_df["SUPPORT"].shift(1)
    return pd.DataFrame({"BREAKOUT_UP": breakout_up.astype(int), "BREAKOUT_DOWN": breakout_down.astype(int)})


def trend_strength(df: pd.DataFrame) -> pd.Series:
    ema_short = ema(df["Close"], span=12)
    ema_long = ema(df["Close"], span=26)
    momentum = ema_short - ema_long
    return momentum
