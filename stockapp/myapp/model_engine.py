"""
model_engine.py — Model definitions, builders, loaders, and ensemble logic.

Supports: LSTM, BiLSTM, GRU, CNN-LSTM (deep learning) and
XGBoost, LightGBM, RandomForest (gradient boosting / tree-based).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ── Optional heavy dependencies ────────────────────────────────────────────
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:  # pragma: no cover
    XGBRegressor = None  # type: ignore[assignment,misc]
    XGBClassifier = None  # type: ignore[assignment,misc]

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMRegressor = None  # type: ignore[assignment,misc]
    LGBMClassifier = None  # type: ignore[assignment,misc]


# ── Model file-name registry ───────────────────────────────────────────────
# Maps canonical model keys → file names on disk.
# IMPORTANT: keys must be lowercase, matching the lookup convention used by
# all calling code (str.lower()).
MODEL_FILE_MAP: dict[str, str] = {
    "lstm":       "lstm.h5",
    "bilstm":     "bilstm.h5",
    "gru":        "gru.h5",
    "cnn_lstm":   "cnn_lstm.h5",   # ← was "cnn-lstm" — fixed KeyError bug
    "xgboost":    "xgboost.pkl",
    "lightgbm":   "lightgbm.pkl",
    "rf":         "rf.pkl",
    "xgb_cls":    "xgb_cls.pkl",   # classification variant
    "lgbm_cls":   "lgbm_cls.pkl",  # classification variant
    "rf_cls":     "rf_cls.pkl",    # classification variant
    "metrics":    "model_metrics.json",
    "scaler":     "scaler.npy",
}

# Kept for backward compatibility
MODEL_NAMES = MODEL_FILE_MAP

# Per-model ensemble weights for regression (must sum to 1.0)
MODEL_WEIGHTS: dict[str, float] = {
    "LSTM":         0.20,
    "BILSTM":       0.20,
    "GRU":          0.20,
    "CNN_LSTM":     0.15,
    "XGBOOST":      0.12,
    "LIGHTGBM":     0.08,
    "RF":           0.05,
}


# ── Keras lazy import ──────────────────────────────────────────────────────
def _import_keras():
    """Lazy Keras import to avoid TensorFlow startup cost at module load time."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Bidirectional,
        Conv1D,
        Dense,
        Dropout,
        GRU,
        LSTM,
        MaxPooling1D,
    )
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    return Sequential, Bidirectional, Conv1D, Dense, Dropout, GRU, LSTM, MaxPooling1D, load_model, Adam


# ── Persistence helpers ────────────────────────────────────────────────────
def save_model_object(obj: Any, path: Path) -> None:
    """Save a Keras model (.h5) or sklearn/xgboost model (.pkl)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".h5":
        if hasattr(obj, "save"):
            obj.save(str(path))
        else:
            joblib.dump(obj, path)
    else:
        joblib.dump(obj, path)
    logger.debug("Saved model artifact: %s", path)


def load_model_object(path: Path) -> Any:
    """Load a Keras (.h5) or sklearn/xgboost (.pkl) model."""
    if path.suffix == ".h5":
        _, _, _, _, _, _, _, _, load_model, _ = _import_keras()
        return load_model(str(path))
    return joblib.load(path)


def load_saved_models(model_dir: Path) -> dict[str, Any]:
    """
    Load all trained model artifacts from *model_dir*.

    Returns a dict mapping canonical model keys (lowercase) to model objects.
    Missing model files are silently skipped.
    """
    loaded: dict[str, Any] = {}
    for key, filename in MODEL_FILE_MAP.items():
        if key in {"metrics", "scaler"}:
            continue
        model_path = model_dir / filename
        if not model_path.exists():
            continue
        try:
            loaded[key] = load_model_object(model_path)
            logger.debug("Loaded model '%s' from %s", key, model_path)
        except Exception as exc:
            logger.warning("Failed to load model '%s': %s", key, exc)
    return loaded


def save_metrics_report(model_dir: Path, metrics: dict[str, Any]) -> None:
    """Persist model metrics to JSON."""
    path = model_dir / MODEL_FILE_MAP["metrics"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved metrics report: %s", path)


def load_metrics_report(model_dir: Path) -> dict[str, Any]:
    """Load persisted model metrics. Returns empty dict if file not found."""
    path = model_dir / MODEL_FILE_MAP["metrics"]
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── ModelLoader dataclass ──────────────────────────────────────────────────
_GLOBAL_MODEL_CACHE: dict[str, Any] | None = None
_GLOBAL_SCALER_CACHE: Any | None = None

@dataclass(frozen=True)
class ModelLoader:
    model_dir: Path

    def get_scaler(self) -> Any:
        global _GLOBAL_SCALER_CACHE
        if _GLOBAL_SCALER_CACHE is not None:
            return _GLOBAL_SCALER_CACHE
            
        scaler_path = self.model_dir / MODEL_FILE_MAP["scaler"]
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}.")
        
        _GLOBAL_SCALER_CACHE = np.load(scaler_path, allow_pickle=True).item()
        return _GLOBAL_SCALER_CACHE

    def get_models(self) -> dict[str, Any]:
        global _GLOBAL_MODEL_CACHE
        if _GLOBAL_MODEL_CACHE is not None:
            return _GLOBAL_MODEL_CACHE
            
        _GLOBAL_MODEL_CACHE = load_saved_models(self.model_dir)
        return _GLOBAL_MODEL_CACHE

    @staticmethod
    def clear_cache() -> None:
        global _GLOBAL_MODEL_CACHE, _GLOBAL_SCALER_CACHE
        _GLOBAL_MODEL_CACHE = None
        _GLOBAL_SCALER_CACHE = None


# ── Deep Learning model builders ──────────────────────────────────────────
def build_lstm_model(time_steps: int, features: int) -> Any:
    """Stacked 2-layer LSTM regression model."""
    Sequential, _, _, Dense, Dropout, _, LSTM, _, _, Adam = _import_keras()
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(time_steps, features)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


def build_bidirectional_lstm_model(time_steps: int, features: int) -> Any:
    """Bidirectional LSTM regression model."""
    Sequential, Bidirectional, _, Dense, Dropout, _, LSTM, _, _, Adam = _import_keras()
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(time_steps, features)),
        Dropout(0.2),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


def build_gru_model(time_steps: int, features: int) -> Any:
    """Stacked 2-layer GRU regression model."""
    Sequential, _, _, Dense, Dropout, GRU, _, _, _, Adam = _import_keras()
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(time_steps, features)),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


def build_cnn_lstm_model(time_steps: int, features: int) -> Any:
    """CNN feature extractor + LSTM temporal model (hybrid)."""
    Sequential, _, Conv1D, Dense, Dropout, _, LSTM, MaxPooling1D, _, Adam = _import_keras()
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same",
               input_shape=(time_steps, features)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


# ── Tree-based model builders ─────────────────────────────────────────────
def build_random_forest(n_estimators: int = 10) -> Any:
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=1)


def build_random_forest_classifier(n_estimators: int = 10) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=1)


def build_xgboost(n_estimators: int = 10) -> Any:
    if XGBRegressor is None:
        raise RuntimeError("XGBoost is not installed. Run: pip install xgboost")
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )


def build_xgboost_classifier(n_estimators: int = 10) -> Any:
    if XGBClassifier is None:
        raise RuntimeError("XGBoost is not installed.")
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        objective="binary:logistic",
        random_state=42,
        verbosity=0,
    )


def build_lightgbm(n_estimators: int = 10) -> Any:
    if LGBMRegressor is None:
        raise RuntimeError("LightGBM is not installed. Run: pip install lightgbm")
    return LGBMRegressor(n_estimators=n_estimators, learning_rate=0.05, random_state=42, verbose=-1)


def build_lightgbm_classifier(n_estimators: int = 10) -> Any:
    if LGBMClassifier is None:
        raise RuntimeError("LightGBM is not installed.")
    return LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, random_state=42, verbose=-1)


# ── Ensemble logic ─────────────────────────────────────────────────────────
def ensemble_predict(predictions: dict[str, float]) -> float:
    """
    Weighted average of model predictions.

    Keys in `predictions` should be UPPERCASE model names matching MODEL_WEIGHTS.
    Unknown models receive zero weight.
    """
    total = 0.0
    weight_sum = 0.0
    for model_key, value in predictions.items():
        weight = MODEL_WEIGHTS.get(model_key.upper(), 0.0)
        total += value * weight
        weight_sum += weight
    if weight_sum <= 0:
        # Fallback: simple mean if no weights matched
        values = list(predictions.values())
        return float(sum(values) / len(values)) if values else 0.0
    return float(total / weight_sum)


def calculate_confidence(predictions: dict[str, float]) -> float:
    """
    Confidence score (0–100) based on inter-model agreement.

    Higher agreement → higher confidence. Uses coefficient of variation
    inverted and scaled to percentage.
    """
    values = np.array(list(predictions.values()), dtype=np.float64)
    if values.size == 0:
        return 0.0
    mean_val = np.mean(np.abs(values))
    if mean_val < 1e-8:
        return 0.0
    cv = np.std(values) / mean_val  # coefficient of variation
    score = (1.0 - min(float(cv), 1.0)) * 100.0
    return float(np.clip(score, 0.0, 100.0))
