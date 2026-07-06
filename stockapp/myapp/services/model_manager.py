"""
model_manager.py — Centralized Singleton for managing Machine Learning Models.

Ensures TensorFlow/Keras and Scikit-Learn models are loaded exactly once in memory.
Configures TensorFlow for CPU-only inference to prevent OOM errors on Render.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Force TensorFlow to use CPU only (crucial for Render free/low-tier instances)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings

# Lazy import to avoid slow startup until needed
def _import_keras_load_model():
    import tensorflow as tf
    # Optional: limit CPU threading to prevent context switching overhead
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    from tensorflow.keras.models import load_model
    return load_model


class ModelManager:
    """
    Singleton class to hold models and scalers in memory.
    """
    _instance = None
    _models: Dict[str, Any] = {}
    _scaler: Any = None
    _is_loaded: bool = False
    _model_dir: Optional[Path] = None

    MODEL_FILE_MAP = {
        "lstm":       "lstm.h5",
        "bilstm":     "bilstm.h5",
        "gru":        "gru.h5",
        "cnn_lstm":   "cnn_lstm.h5",
        "xgboost":    "xgboost.pkl",
        "lightgbm":   "lightgbm.pkl",
        "rf":         "rf.pkl",
        "xgb_cls":    "xgb_cls.pkl",
        "lgbm_cls":   "lgbm_cls.pkl",
        "rf_cls":     "rf_cls.pkl",
        "metrics":    "model_metrics.json",
        "scaler":     "scaler.npy",
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, model_dir: Path):
        cls._model_dir = model_dir
        # Lazy loading: we do NOT call _load_all_artifacts() here.
        # It will be called inside get_models() or get_scaler() on the first request.

    @classmethod
    def _load_all_artifacts(cls):
        """Loads the scaler and all available models into memory."""
        logger.info("Initializing ModelManager: Loading ML artifacts into memory...")
        
        if cls._model_dir is None or not cls._model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {cls._model_dir}")
        
        missing_files = []
        
        # Load scaler
        scaler_path = cls._model_dir / cls.MODEL_FILE_MAP["scaler"]
        if scaler_path.exists():
            cls._scaler = np.load(scaler_path, allow_pickle=True).item()
            logger.info("Scaler loaded successfully.")
        else:
            logger.warning("Scaler artifact not found at %s.", scaler_path)
            missing_files.append(cls.MODEL_FILE_MAP["scaler"])

        # Load models
        loaded_count = 0
        for key, filename in cls.MODEL_FILE_MAP.items():
            if key in {"metrics", "scaler"}:
                continue
            
            model_path = cls._model_dir / filename
            if not model_path.exists():
                missing_files.append(filename)
                continue
            
            try:
                if model_path.suffix == ".h5":
                    load_model = _import_keras_load_model()
                    cls._models[key] = load_model(str(model_path))
                else:
                    cls._models[key] = joblib.load(model_path)
                logger.info("Loaded model '%s'", key)
                loaded_count += 1
            except Exception as exc:
                logger.warning("Failed to load model '%s': %s", key, exc)
                missing_files.append(f"{filename} (corrupted)")
                
        if loaded_count == 0 and missing_files:
            error_msg = f"No models could be loaded. Missing or corrupted files: {', '.join(missing_files)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        cls._is_loaded = True
        logger.info("ModelManager initialization complete.")

    @classmethod
    def get_models(cls) -> Dict[str, Any]:
        """Returns the dictionary of loaded models."""
        if not cls._is_loaded:
            if cls._model_dir is None:
                raise RuntimeError("ModelManager not initialized with a model_dir.")
            cls._load_all_artifacts()
        return cls._models

    @classmethod
    def get_scaler(cls) -> Any:
        """Returns the loaded scaler."""
        if not cls._is_loaded:
            if cls._model_dir is None:
                raise RuntimeError("ModelManager not initialized with a model_dir.")
            cls._load_all_artifacts()
        if cls._scaler is None:
            raise FileNotFoundError("Scaler artifact is missing from memory.")
        return cls._scaler

    @classmethod
    def clear_cache(cls) -> None:
        """Clears models from memory."""
        cls._models = {}
        cls._scaler = None
        cls._is_loaded = False
        import gc
        gc.collect()
        logger.info("ModelManager cache cleared.")
