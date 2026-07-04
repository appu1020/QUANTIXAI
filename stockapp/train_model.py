import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor

from myapp.feature_engineering import create_sequences, engineer_features
from myapp.ml_models import (MODEL_NAMES, build_cnn_lstm_model,
                             build_gru_model, build_transformer_model,
                             compute_metrics, save_metrics_report,
                             save_model_object)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "cnn_lstm.h5"
SCALER_PATH = MODEL_DIR / "scaler.npy"


def download_price_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    df = hist[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        raise ValueError(f"No historical data found for {symbol} with period={period}")
    return df


def train(symbol: str, period: str = "2y", epochs: int = 20, batch_size: int = 32, time_steps: int = 60):
    raw_df = download_price_data(symbol, period)
    engineered_df = engineer_features(raw_df)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(engineered_df.values)
    target = raw_df["Close"].values

    X, y = create_sequences(scaled_features, target, time_steps=time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model_report = {}

    keras_models = {
        "cnn_lstm": build_cnn_lstm_model(time_steps=time_steps, features=X.shape[2]),
        "gru": build_gru_model(time_steps=time_steps, features=X.shape[2]),
        "transformer": build_transformer_model(time_steps=time_steps, features=X.shape[2]),
    }

    for model_name, model in keras_models.items():
        callback = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[callback],
            verbose=1,
        )
        preds = model.predict(X_test)
        model_report[model_name] = compute_metrics(y_test, preds)
        save_model_object(model, MODEL_DIR / MODEL_NAMES[model_name])

    flattened_train = X_train.reshape((X_train.shape[0], -1))
    flattened_test = X_test.reshape((X_test.shape[0], -1))

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, objective="reg:squarederror", random_state=42, verbosity=0)
    xgb.fit(flattened_train, y_train)
    xgb_preds = xgb.predict(flattened_test)
    model_report["xgboost"] = compute_metrics(y_test, xgb_preds)
    save_model_object(xgb, MODEL_DIR / MODEL_NAMES["xgboost"])

    lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    lgbm.fit(flattened_train, y_train)
    lgbm_preds = lgbm.predict(flattened_test)
    model_report["lightgbm"] = compute_metrics(y_test, lgbm_preds)
    save_model_object(lgbm, MODEL_DIR / MODEL_NAMES["lightgbm"])

    np.save(SCALER_PATH, scaler, allow_pickle=True)
    save_metrics_report(MODEL_DIR, model_report)

    print(f"Training complete for {symbol}.")
    print(json.dumps(model_report, indent=2))
    print(f"Saved trained model artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline train ensemble models for stock prediction.")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol to train on")
    parser.add_argument("--period", type=str, default="2y", help="Historical period for training data")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for training")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    train(symbol=args.symbol, period=args.period, epochs=args.epochs, batch_size=args.batch_size)
