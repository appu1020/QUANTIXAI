"""
training_engine.py — Model training pipeline.

Trains regression models (price prediction) and optionally classification
models (directional BUY/SELL prediction). Persists artifacts and reports
metrics to the database.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

def _get_tf_callbacks():
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    return EarlyStopping, ReduceLROnPlateau

from .data_pipeline import build_feature_dataset
from .feature_engineering import (
    add_classification_target,
    create_classification_sequences,
    create_sequences,
)
from .model_engine import (
    MODEL_FILE_MAP,
    LGBMClassifier,
    LGBMRegressor,
    XGBClassifier,
    XGBRegressor,
    build_bidirectional_lstm_model,
    build_cnn_lstm_model,
    build_gru_model,
    build_lightgbm,
    build_lightgbm_classifier,
    build_lstm_model,
    build_random_forest,
    build_random_forest_classifier,
    build_xgboost,
    build_xgboost_classifier,
    save_metrics_report,
    save_model_object,
)

logger = logging.getLogger(__name__)

# Canonical display names → file map keys
_KERAS_MODEL_MAP: dict[str, str] = {
    "LSTM":     "lstm",
    "BiLSTM":   "bilstm",
    "GRU":      "gru",
    "CNN-LSTM": "cnn_lstm",  # Fixed: was "cnn-lstm" → KeyError
}


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    
    # Calculate directional accuracy (trend matching)
    if len(y_true) > 1:
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        dir_acc = float(np.mean((true_diff > 0) == (pred_diff > 0)))
    else:
        dir_acc = 0.0
        
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100),
        "r2":   float(r2_score(y_true, y_pred)),
        "accuracy": dir_acc,
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_models(
    symbol: str,
    interval: str = "1d",
    time_steps: int = 60,
    epochs: int = 30,
    batch_size: int = 32,
    model_dir: Path | None = None,
    train_classifiers: bool = True,
) -> dict[str, Any]:
    """
    Full training pipeline: downloads data, engineers features, trains all
    regression and classification models, persists artifacts, and returns a
    metrics report dict.

    Args:
        symbol:             Ticker symbol (e.g. 'AAPL').
        interval:           Data interval ('1d', '1h', '15m', '5m').
        time_steps:         LSTM/GRU sequence look-back window.
        epochs:             Max training epochs for deep learning models.
        batch_size:         Mini-batch size for deep learning.
        model_dir:          Directory to save model artifacts.
        train_classifiers:  Whether to also train directional classifiers.

    Returns:
        Nested dict: { model_name: { metric_name: value } }.
    """
    model_dir = Path(model_dir or Path(__file__).resolve().parent / "models")
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training pipeline for %s @ %s", symbol, interval)

    # ── 1. Data preparation ───────────────────────────────────────────────
    df = build_feature_dataset(symbol=symbol, interval=interval)
    logger.info("Feature dataset shape: %s", df.shape)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df.values)

    # Regression target: Next-close price scaled WITH the same scaler as features.
    # FIX V3: Previously target was raw (unscaled) while features were scaled,
    # causing LSTM/GRU/BiLSTM to learn a mapping from [0,1] features → raw prices.
    # Now we scale the target using the Close column's transform from the fitted scaler.
    from .feature_engineering import FEATURE_COLUMNS
    close_idx = FEATURE_COLUMNS.index("Close")
    close_min = float(scaler.data_min_[close_idx])
    close_max = float(scaler.data_max_[close_idx])
    close_range = max(close_max - close_min, 1e-8)

    raw_close_target = df["Close"].values
    # Scale close target to [0,1] matching how the scaler transforms Close
    scaled_close_target = (raw_close_target - close_min) / close_range

    # Classification target: 1 if price goes up next candle, 0 otherwise
    df_with_cls = add_classification_target(df, horizon=1)
    direction_target = df_with_cls["TARGET_DIRECTION"].fillna(0).values

    # Deep learning sequences use SCALED target (FIX V3)
    X, y_reg = create_sequences(scaled_features, scaled_close_target, time_steps=time_steps)
    # Tree models use raw prices (they handle arbitrary scale)
    _, y_reg_raw = create_sequences(scaled_features, raw_close_target, time_steps=time_steps)
    X_cls, y_cls = create_classification_sequences(scaled_features, direction_target, time_steps=time_steps)

    # Time-series split — no shuffle
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, shuffle=False)
    _, _, y_raw_train, y_raw_test = train_test_split(X, y_reg_raw, test_size=0.2, shuffle=False)
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
        X_cls, y_cls, test_size=0.2, shuffle=False
    )

    report: dict[str, Any] = {}

    EarlyStopping, ReduceLROnPlateau = _get_tf_callbacks()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=0, min_lr=1e-5),
    ]

    # ── 2. Deep learning regression models ───────────────────────────────
    keras_builders = {
        "LSTM":     build_lstm_model,
        "BiLSTM":   build_bidirectional_lstm_model,
        "GRU":      build_gru_model,
        "CNN-LSTM": build_cnn_lstm_model,
    }

    for display_name, builder_fn in keras_builders.items():
        file_key = _KERAS_MODEL_MAP[display_name]  # e.g. "cnn_lstm"
        logger.info("Training %s…", display_name)
        try:
            model = builder_fn(time_steps=time_steps, features=X.shape[2])
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )
            preds = model.predict(X_test, verbose=0).reshape(-1)
            report[display_name] = compute_regression_metrics(y_test, preds)
            save_model_object(model, model_dir / MODEL_FILE_MAP[file_key])
            logger.info("%s metrics: %s", display_name, report[display_name])
        except Exception as exc:
            logger.error("Failed to train %s: %s", display_name, exc)

    # ── 3. Tree-based regression models ───────────────────────────────────
    flat_train = X_train.reshape(X_train.shape[0], -1)
    flat_test  = X_test.reshape(X_test.shape[0], -1)

    logger.info("Training RandomForest (regression)…")
    rf = build_random_forest()
    rf.fit(flat_train, y_raw_train)
    report["RandomForest"] = compute_regression_metrics(y_raw_test, rf.predict(flat_test))
    save_model_object(rf, model_dir / MODEL_FILE_MAP["rf"])

    if XGBRegressor is not None:
        logger.info("Training XGBoost (regression)…")
        xgb = build_xgboost()
        xgb.fit(flat_train, y_raw_train)
        report["XGBoost"] = compute_regression_metrics(y_raw_test, xgb.predict(flat_test))
        save_model_object(xgb, model_dir / MODEL_FILE_MAP["xgboost"])

    if LGBMRegressor is not None:
        logger.info("Training LightGBM (regression)…")
        lgbm = build_lightgbm()
        lgbm.fit(flat_train, y_raw_train)
        report["LightGBM"] = compute_regression_metrics(y_raw_test, lgbm.predict(flat_test))
        save_model_object(lgbm, model_dir / MODEL_FILE_MAP["lightgbm"])

    # ── 4. Classification models (directional) ────────────────────────────
    if train_classifiers:
        flat_cls_train = X_cls_train.reshape(X_cls_train.shape[0], -1)
        flat_cls_test  = X_cls_test.reshape(X_cls_test.shape[0], -1)

        logger.info("Training RandomForest (classifier)…")
        rf_cls = build_random_forest_classifier()
        rf_cls.fit(flat_cls_train, y_cls_train)
        report["RandomForest_Classifier"] = compute_classification_metrics(
            y_cls_test, rf_cls.predict(flat_cls_test)
        )
        save_model_object(rf_cls, model_dir / MODEL_FILE_MAP["rf_cls"])

        if XGBClassifier is not None:
            logger.info("Training XGBoost (classifier)…")
            xgb_cls = build_xgboost_classifier()
            xgb_cls.fit(flat_cls_train, y_cls_train)
            report["XGBoost_Classifier"] = compute_classification_metrics(
                y_cls_test, xgb_cls.predict(flat_cls_test)
            )
            save_model_object(xgb_cls, model_dir / MODEL_FILE_MAP["xgb_cls"])

        if LGBMClassifier is not None:
            logger.info("Training LightGBM (classifier)…")
            lgbm_cls = build_lightgbm_classifier()
            lgbm_cls.fit(flat_cls_train, y_cls_train)
            report["LightGBM_Classifier"] = compute_classification_metrics(
                y_cls_test, lgbm_cls.predict(flat_cls_test)
            )
            save_model_object(lgbm_cls, model_dir / MODEL_FILE_MAP["lgbm_cls"])

    # ── 5. Persist scaler and metrics ─────────────────────────────────────
    np.save(model_dir / MODEL_FILE_MAP["scaler"], scaler, allow_pickle=True)
    save_metrics_report(model_dir, report)

    # ── 6. Persist to DB ──────────────────────────────────────────────────
    _persist_training_metrics(symbol=symbol, interval=interval, report=report)

    # Clear the global model cache so the next request loads the new models
    from .model_engine import ModelLoader
    ModelLoader.clear_cache()

    logger.info("Training complete. Models saved to: %s", model_dir)
    return report


def _persist_training_metrics(symbol: str, interval: str, report: dict[str, Any]) -> None:
    """Save training metrics to TrainingMetric DB model (best-effort)."""
    try:
        from .models import TrainingMetric
        for model_name, metrics in report.items():
            family = "classification" if "Classifier" in model_name else "regression"
            TrainingMetric.objects.create(
                model_name=model_name,
                model_family=family,
                symbol=symbol,
                interval=interval,
                metrics=metrics,
            )
    except Exception as exc:
        logger.warning("Could not persist training metrics to DB: %s", exc)
