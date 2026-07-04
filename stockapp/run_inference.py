import json
from pathlib import Path

import numpy as np
import yfinance as yf

from myapp.ml_models import load_saved_models, ensemble_predict, calculate_confidence, load_metrics_report
from myapp.feature_engineering import prepare_inference_window

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
SCALER_PATH = MODEL_DIR / "scaler.npy"

symbol = "AAPL"

if not SCALER_PATH.exists():
    raise FileNotFoundError("Scaler not found. Run training first.")

scaler = np.load(SCALER_PATH, allow_pickle=True).item()
trained_models = load_saved_models(MODEL_DIR)
if not trained_models:
    raise FileNotFoundError("No trained models found in models directory.")

stock = yf.Ticker(symbol)
hist = stock.history(period="2y")
if hist.empty or len(hist) < 60:
    raise ValueError("Not enough historical data for inference")

df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
X_input, features_df = prepare_inference_window(df, scaler, time_steps=60)

predictions = {}
for model_name, model in trained_models.items():
    try:
        if model_name in ["xgboost", "lightgbm"]:
            pred = model.predict(X_input.reshape(1, -1))
            val = float(pred.reshape(-1)[0]) if hasattr(pred, "reshape") else float(pred[0])
        else:
            pred = model.predict(X_input, verbose=0)
            val = float(np.array(pred).reshape(-1)[0])
        predictions[model_name] = val
    except Exception as e:
        predictions[model_name] = None
        print(f"Prediction failed for {model_name}: {e}")
# Map internal model keys to the human-readable names used by the ensemble weights
key_mapping = {
    "cnn_lstm": "CNN-LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
mapped_preds = {key_mapping.get(k, k): v for k, v in predictions.items() if v is not None}

ensemble_price = ensemble_predict(mapped_preds)
confidence = calculate_confidence(mapped_preds)
metrics = load_metrics_report(MODEL_DIR)

out = {
    "symbol": symbol,
    "predictions": predictions,
    "ensemble_price": ensemble_price,
    "confidence": confidence,
    "metrics": metrics,
}
print(json.dumps(out, indent=2))
