import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, Input,
                                     LayerNormalization, LSTM, MaxPooling1D,
                                     MultiHeadAttention)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

MODEL_NAMES = {
    "cnn_lstm": "cnn_lstm.h5",
    "gru": "gru.h5",
    "transformer": "transformer.h5",
    "xgboost": "xgboost.pkl",
    "lightgbm": "lightgbm.pkl",
    "metrics": "model_metrics.json",
}


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
    }


def build_cnn_lstm_model(time_steps: int, features: int):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same", input_shape=(time_steps, features)),
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


def build_gru_model(time_steps: int, features: int):
    from tensorflow.keras.layers import GRU

    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same", input_shape=(time_steps, features)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


def build_transformer_model(time_steps: int, features: int):
    inputs = Input(shape=(time_steps, features))
    x = Dense(64, activation="relu")(inputs)
    attention = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = LayerNormalization()(x + attention)
    x = Dense(64, activation="relu")(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model


def flatten_sequences(features: np.ndarray, time_steps: int):
    flattened = []
    for i in range(len(features) - time_steps):
        flattened.append(features[i : i + time_steps].reshape(-1))
    return np.array(flattened)


def save_model_object(obj, path: Path):
    if path.suffix == ".h5":
        obj.save(path)
    else:
        joblib.dump(obj, path)


def load_model_object(path: Path):
    if path.suffix == ".h5":
        return load_model(path)
    return joblib.load(path)


def load_saved_models(model_dir: Path):
    loaded = {}
    for name, filename in MODEL_NAMES.items():
        if name == "metrics":
            continue
        model_path = model_dir / filename
        if model_path.exists():
            try:
                loaded[name] = load_model_object(model_path)
            except Exception:
                try:
                    loaded[name] = joblib.load(model_path)
                except Exception:
                    continue
    return loaded


def load_metrics_report(model_dir: Path):
    metrics_path = model_dir / MODEL_NAMES["metrics"]
    if not metrics_path.exists():
        return {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def save_metrics_report(model_dir: Path, report: dict):
    metrics_path = model_dir / MODEL_NAMES["metrics"]
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def ensemble_predict(predictions: dict):
    weights = {
        "CNN-LSTM": 0.22,
        "GRU": 0.22,
        "XGBoost": 0.18,
        "Transformer": 0.28,
        "LightGBM": 0.10,
    }
    total, weight_sum = 0.0, 0.0
    for model_name, value in predictions.items():
        model_weight = weights.get(model_name, 0.0)
        total += value * model_weight
        weight_sum += model_weight
    if weight_sum <= 0:
        raise ValueError("Ensemble weights sum to zero")
    return float(total / weight_sum)


def calculate_confidence(predictions: dict):
    values = np.array(list(predictions.values()), dtype=np.float32)
    mean_value = np.mean(np.abs(values))
    if mean_value < 1e-8:
        return 0.0
    variance = np.std(values) / mean_value
    score = (1.0 - min(max(variance, 0.0), 1.0)) * 100.0
    return float(max(min(score, 100.0), 0.0))
