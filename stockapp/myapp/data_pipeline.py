"""
data_pipeline.py — Market data acquisition, validation, and persistence.

Supports multi-timeframe data collection (5m, 15m, 1h, 1d) from yfinance,
with automated validation, outlier detection, and optional DB caching.

FIXES APPLIED:
  M1/M6: Added TTL-based in-memory cache. yFinance aggressively disk-caches;
          our cache layer controls staleness. 30s for intraday, 5min for daily.
  M5:    Added fetch_live_quote() using yf.Ticker.fast_info — sub-100ms latency
          vs the previous download_market_data() which fetched 5y of history.
  Added: retry logic with exponential backoff (3 attempts).
  Added: detailed logging for every API call outcome.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

# ── Timeframe configuration ────────────────────────────────────────────────
TIMEFRAME_CONFIG: dict[str, dict[str, Any]] = {
    "5m":  {"period": "7d",   "interval": "5m",  "cache_ttl": 30},   # 30s cache
    "15m": {"period": "60d",  "interval": "15m", "cache_ttl": 60},   # 1min cache
    "1h":  {"period": "180d", "interval": "60m", "cache_ttl": 120},  # 2min cache
    "1d":  {"period": "5y",   "interval": "1d",  "cache_ttl": 300},  # 5min cache
}

REQUIRED_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]

# ── In-memory TTL cache (FIX M1/M6) ──────────────────────────────────────────
_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = Lock()


def _cache_key(symbol: str, interval: str) -> str:
    return f"{symbol.upper()}_{interval}"


def _cache_get(key: str, ttl: int) -> pd.DataFrame | None:
    """Return cached DataFrame if it exists and is not stale, else None."""
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        age = time.monotonic() - entry["ts"]
        if age > ttl:
            logger.debug("Cache MISS (stale, age=%.0fs): %s", age, key)
            return None
        logger.debug("Cache HIT (age=%.0fs): %s", age, key)
        return entry["df"]


def _cache_put(key: str, df: pd.DataFrame) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = {"df": df, "ts": time.monotonic()}


def invalidate_cache(symbol: str | None = None, interval: str | None = None) -> None:
    """Invalidate cache for a specific symbol/interval or everything."""
    with _CACHE_LOCK:
        if symbol is None and interval is None:
            _CACHE.clear()
            logger.info("All market data cache cleared")
            return
        to_delete = []
        for key in list(_CACHE.keys()):
            if symbol and symbol.upper() not in key:
                continue
            if interval and interval not in key:
                continue
            to_delete.append(key)
        for key in to_delete:
            del _CACHE[key]
        if to_delete:
            logger.info("Cache invalidated for keys: %s", to_delete)


# ── Live quote (FIX M5) ───────────────────────────────────────────────────────
def fetch_live_quote(symbol: str) -> dict[str, Any]:
    """
    FIX M5: Fast live price fetch using yfinance fast_info or a 2-day history.

    This replaces the full historical download (5y) just to get .iloc[-1].
    Latency: typically < 300ms vs 3-10s for the old approach.

    Returns:
        {
            "symbol": str, "close": float, "open": float, "high": float,
            "low": float, "volume": int, "change": float, "change_pct": float,
            "timestamp": str, "source": str
        }
    """
    cache_key = f"live_{symbol.upper()}"
    with _CACHE_LOCK:
        entry = _CACHE.get(cache_key)
        if entry and (time.monotonic() - entry["ts"]) < 10:  # 10s cache for live quotes
            logger.debug("Live quote cache HIT: %s", symbol)
            return entry["df"]  # Actually a dict here

    t0 = time.monotonic()
    result: dict[str, Any] = {
        "symbol": symbol.upper(),
        "close": 0.0, "open": 0.0, "high": 0.0, "low": 0.0,
        "volume": 0, "change": 0.0, "change_pct": 0.0,
        "timestamp": "", "source": "yfinance",
    }

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)

        # Strategy 1: 1-minute history (absolute latest price)
        # Replaces fast_info to guarantee accurate real-time data as requested
        hist = ticker.history(period="5d", interval="1m")
        if not hist.empty:
            latest = hist.iloc[-1]
            # To calculate change_pct, we need previous day's close.
            # We can get that from info or fallback to a 1d history fetch if really needed.
            # But the most reliable way without info is history(period="2d", interval="1d").
            # However, for performance, we can compute change from the first candle of the day if we only have 1m data,
            # or we can do a quick 2d 1d fetch to get exact prev close. Let's do a quick 1d fetch for prev_close if needed,
            # or just rely on the 5d 1m data and find yesterday's last close.
            
            # Find yesterday's close by looking for the last candle of the previous day
            current_day = hist.index[-1].date()
            prev_day_data = hist[hist.index.date < current_day]
            prev_close = float(prev_day_data["Close"].iloc[-1]) if len(prev_day_data) > 0 else float(latest["Close"])
            
            close = float(latest["Close"])
            change = close - prev_close
            change_pct = (change / max(prev_close, 1e-8)) * 100
            
            result.update({
                "close":      round(close, 4),
                "open":       round(float(latest["Open"]), 4),
                "high":       round(float(latest["High"]), 4),
                "low":        round(float(latest["Low"]), 4),
                "volume":     int(latest["Volume"]),
                "change":     round(change, 4),
                "change_pct": round(change_pct, 4),
                "timestamp":  str(hist.index[-1]),
                "source":     "yfinance_1m_history",
            })
            logger.info(
                "Live quote [1m_history] %s: $%.2f (%+.2f%%) in %.0fms",
                symbol, close, change_pct, (time.monotonic() - t0) * 1000,
            )
            # Store in cache with implied TTL
            _cache_put(cache_key, result)
            return result
        else:
            logger.warning("No 1m data returned for %s", symbol)

    except Exception as exc:
        logger.error("fetch_live_quote failed for %s: %s", symbol, exc)

    return result


def fetch_live_quotes_batch(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """
    Batch live price fetch for multiple symbols (e.g., portfolio/watchlist).
    FIX M7: Uses a single yfinance download() call for all symbols at once.
    """
    if not symbols:
        return {}

    results: dict[str, dict[str, Any]] = {}
    try:
        import yfinance as yf
        t0 = time.monotonic()
        # Single multi-symbol download — much faster than serial calls
        data = yf.download(
            tickers=" ".join(symbols),
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        elapsed = (time.monotonic() - t0) * 1000
        logger.info("Batch download for %d symbols in %.0fms", len(symbols), elapsed)

        for sym in symbols:
            try:
                if not isinstance(data, pd.DataFrame):
                    results[sym] = _empty_quote(sym)
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    # Safely handle multi-index columns (yfinance 0.2+ with group_by="ticker")
                    df = data[sym] if sym in data.columns.get_level_values(0) else pd.DataFrame()
                else:
                    # Single level index (older yfinance versions or single ticker without group_by)
                    df = data

                if df.empty or "Close" not in df.columns:
                    results[sym] = _empty_quote(sym)
                    continue

                df = df.dropna(subset=["Close"])
                if df.empty:
                    results[sym] = _empty_quote(sym)
                    continue
                
                # Find yesterday's close for accurate change_pct
                current_day = df.index[-1].date()
                prev_day_data = df[df.index.date < current_day]
                prev = float(prev_day_data["Close"].iloc[-1]) if len(prev_day_data) > 0 else float(df["Close"].iloc[-1])

                close = float(df["Close"].iloc[-1])
                change = close - prev
                change_pct = (change / max(prev, 1e-8)) * 100
                results[sym] = {
                    "symbol":     sym,
                    "close":      round(close, 4),
                    "open":       round(float(df["Open"].iloc[-1]) if "Open" in df.columns else close, 4),
                    "high":       round(float(df["High"].iloc[-1]) if "High" in df.columns else close, 4),
                    "low":        round(float(df["Low"].iloc[-1]) if "Low" in df.columns else close, 4),
                    "volume":     int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0,
                    "change":     round(change, 4),
                    "change_pct": round(change_pct, 4),
                    "timestamp":  str(df.index[-1]),
                    "source":     "yfinance_batch_1m",
                }
            except Exception as e:
                logger.warning("Batch quote parse failed for %s: %s", sym, e)
                results[sym] = _empty_quote(sym)

    except Exception as exc:
        logger.error("Batch download failed: %s. Falling back to serial.", exc)
        for sym in symbols:
            results[sym] = fetch_live_quote(sym)

    return results


def _empty_quote(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol, "close": 0.0, "open": 0.0,
        "high": 0.0, "low": 0.0, "volume": 0,
        "change": 0.0, "change_pct": 0.0, "timestamp": "", "source": "error",
    }


# ── Download with TTL cache and retry (FIX M1/M6) ────────────────────────────
def download_market_data(
    symbol: str,
    interval: str = "1d",
    force_fresh: bool = False,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Download OHLCV data from yfinance for the given symbol and interval.

    FIX M1/M6: Added TTL-based caching (30s-5min depending on interval).
    force_fresh=True bypasses cache for live-data endpoint.

    Returns a cleaned, deduplicated DataFrame indexed by DatetimeIndex.
    Raises ValueError if data is empty or malformed.
    """
    if interval not in TIMEFRAME_CONFIG:
        raise ValueError(
            f"Unsupported interval '{interval}'. Supported: {sorted(TIMEFRAME_CONFIG)}"
        )

    config = TIMEFRAME_CONFIG[interval]
    cache_ttl = config["cache_ttl"]
    key = _cache_key(symbol, interval)

    if not force_fresh:
        cached = _cache_get(key, ttl=cache_ttl)
        if cached is not None:
            return cached

    logger.info(
        "Downloading %s | interval=%s | period=%s (cache_ttl=%ds)",
        symbol, interval, config["period"], cache_ttl,
    )

    df = pd.DataFrame()
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                period=config["period"],
                interval=config["interval"],
                auto_adjust=True,
            )
            if not df.empty:
                break
            logger.warning(
                "yFinance returned empty DataFrame for %s @ %s (attempt %d/%d)",
                symbol, interval, attempt, retries,
            )
        except Exception as exc:
            last_exc = exc
            wait = 2 ** (attempt - 1)
            logger.warning(
                "yFinance request failed for %s @ %s (attempt %d/%d): %s. Retrying in %ds.",
                symbol, interval, attempt, retries, exc, wait,
            )
            if attempt < retries:
                time.sleep(wait)

    if df.empty:
        msg = f"yfinance returned empty data for {symbol} at interval {interval} after {retries} attempts."
        if last_exc:
            msg += f" Last error: {last_exc}"
        raise ValueError(msg)

    # Flatten multi-level columns that yfinance sometimes produces
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Downloaded data missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(how="any")
    df = df.astype(float)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if df.empty:
        raise ValueError(f"No valid rows for {symbol} @ {interval} after cleaning.")

    logger.info(
        "Downloaded %s @ %s: %d rows, latest close=$%.2f @ %s",
        symbol, interval, len(df),
        float(df["Close"].iloc[-1]), str(df.index[-1]),
    )

    _cache_put(key, df)
    return df


# ── Validation ─────────────────────────────────────────────────────────────
def validate_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Assert structural integrity of a market data DataFrame."""
    if df.empty:
        raise ValueError("Market data is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Market data index must be a DatetimeIndex.")
    if df[REQUIRED_COLUMNS].isna().any().any():
        counts = df[REQUIRED_COLUMNS].isna().sum().to_dict()
        raise ValueError(f"Market data contains missing values: {counts}")
    duplicates = df.index.duplicated().sum()
    if duplicates:
        logger.warning("%d duplicate timestamps found for market data.", duplicates)
    return df


# ── Outlier detection ──────────────────────────────────────────────────────
def detect_outliers(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """Return rows where Close price is an extreme outlier (IQR-based)."""
    price = df["Close"]
    q1, q3 = price.quantile(0.25), price.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
    return df[(price < lower) | (price > upper)]


# ── Cleaning ───────────────────────────────────────────────────────────────
def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, deduplicate, and forward/backward fill a market DataFrame."""
    df = df.copy().sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS].ffill().bfill()
    return df.dropna(how="any")


# ── Orchestrated pipeline ──────────────────────────────────────────────────
def harmonize_market_data(symbol: str, interval: str = "1d", force_fresh: bool = False) -> pd.DataFrame:
    """Full download → validate → clean → outlier-remove pipeline."""
    df = download_market_data(symbol, interval, force_fresh=force_fresh)
    df = validate_market_data(df)
    df = clean_market_data(df)

    outliers = detect_outliers(df)
    if not outliers.empty:
        logger.debug(
            "Removing %d outlier rows for %s @ %s", len(outliers), symbol, interval
        )
        df = df[~df.index.isin(outliers.index)]
        df = clean_market_data(df)

    return df


def build_feature_dataset(symbol: str, interval: str = "1d") -> pd.DataFrame:
    """Download, clean, and engineer features for model training."""
    from .feature_engineering import engineer_features
    df = harmonize_market_data(symbol, interval)
    features = engineer_features(df)
    if features.isna().any().any():
        raise ValueError("Feature dataset contains NaN values after engineering.")
    return features


# ── DB persistence helpers ─────────────────────────────────────────────────
def persist_to_db(symbol: str, interval: str, df: pd.DataFrame) -> int:
    """Upsert OHLCV rows into the HistoricalMarketData model."""
    try:
        from .models import HistoricalMarketData
    except ImportError:
        logger.warning("Django not configured — skipping DB persistence.")
        return 0

    created = 0
    for ts, row in df.iterrows():
        _, was_created = HistoricalMarketData.objects.update_or_create(
            symbol=symbol,
            interval=interval,
            timestamp=ts,
            defaults={
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": float(row["Volume"]),
                "source": "yfinance",
            },
        )
        if was_created:
            created += 1

    logger.info("Persisted %d new rows for %s @ %s", created, symbol, interval)
    return created


def load_from_db(symbol: str, interval: str) -> pd.DataFrame:
    """Load OHLCV data from the HistoricalMarketData DB model."""
    try:
        from .models import HistoricalMarketData
    except ImportError:
        return pd.DataFrame()

    qs = HistoricalMarketData.objects.filter(
        symbol=symbol, interval=interval
    ).values("timestamp", "open", "high", "low", "close", "volume")

    if not qs.exists():
        return pd.DataFrame()

    df = pd.DataFrame.from_records(qs)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    df = df.set_index("timestamp").sort_index()
    df.index = pd.to_datetime(df.index)
    return df.astype(float)


def fetch_candle_data(symbol: str, interval: str = "1d") -> list[dict[str, Any]]:
    """
    Return OHLCV data as a list of dicts compatible with TradingView Lightweight Charts.

    Format: [{"time": unix_seconds or "YYYY-MM-DD", "open": x, ...}]
    """
    df = harmonize_market_data(symbol, interval)
    result: list[dict[str, Any]] = []
    for ts, row in df.iterrows():
        if interval == "1d":
            time_val = pd.Timestamp(ts).strftime("%Y-%m-%d")
        else:
            time_val = int(pd.Timestamp(ts).timestamp())

        result.append({
            "time":   time_val,
            "open":   round(float(row["Open"]), 4),
            "high":   round(float(row["High"]), 4),
            "low":    round(float(row["Low"]), 4),
            "close":  round(float(row["Close"]), 4),
            "volume": round(float(row["Volume"]), 0),
        })
    return result
