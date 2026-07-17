import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

import api.run_lstm_model_inference as lstm
import api.run_qlstm_model_inference as qlstm
import api.run_custom_model_inference as custom
import api.run_hybrid1_model_inference as hybrid1
import api.run_hybrid2_model_inference as hybrid2

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score
)
app = FastAPI()

WINDOW_DAYS = 10
CURRENT_DIR = Path(__file__).parent

def load_model_weights(model_file_name: str) -> dict:
  file_path = CURRENT_DIR / "weights" / model_file_name
  if not file_path.exists():
    print(f"Warning: Artifact {model_file_name} not found.")
    return None
  return np.load(file_path, allow_pickle=True).item()

WEIGHTS = {
  "lstm": load_model_weights("lstm_weights.npy"),
  "qlstm": load_model_weights("qlstm_weights.npy"),
  "custom_qnn": load_model_weights("custom_qnn_weights.npy"),
  "hybrid_qnn1": load_model_weights("hybrid_qnn1_weights.npy"),
  "hybrid_qnn2": load_model_weights("hybrid_qnn2_weights.npy"),
  "hybrid_qnn1_binary": load_model_weights("hybrid_qnn1_binary_weights.npy"),
  "hybrid_qnn2_binary": load_model_weights("hybrid_qnn2_binary_weights.npy"),
}

def load_preprocessor(preprocessor_file_name: str):
  try:
    JOBLIB_PATH = CURRENT_DIR / "preprocessors" / preprocessor_file_name
    preprocessor = joblib.load(JOBLIB_PATH)

    FEATURE_CANDIDATES = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi_14",
        "macd",
        "macd_signal",
        "adx_14",
    )
    default_candidate_features = [
        feature for feature in FEATURE_CANDIDATES
    ]
    
    if isinstance(preprocessor, dict):
      selector = preprocessor.get("selector")
      x_scaler = preprocessor.get("x_scaler")
      y_scaler = preprocessor.get("y_scaler")
      selected_features = preprocessor.get("selected_features")
      candidate_features = preprocessor.get("candidate_features", default_candidate_features)
      lookback = preprocessor.get("lookback")
      feature_range = preprocessor.get("feature_range")
      sequence_length = preprocessor.get("lookback", WINDOW_DAYS) 
    else:
      selector = getattr(preprocessor, "selector", None)
      x_scaler = getattr(preprocessor, "x_scaler", None)
      y_scaler = getattr(preprocessor, "y_scaler", None)
      selected_features = getattr(preprocessor, "selected_features", None)
      candidate_features = getattr(preprocessor, "candidate_features", default_candidate_features)
      feature_range = getattr(preprocessor, "feature_range", None)
      lookback = getattr(preprocessor, "lookback", None)
      sequence_length = getattr(preprocessor, "lookback", WINDOW_DAYS)
    return selector, x_scaler, y_scaler, selected_features, candidate_features, lookback, feature_range, sequence_length
  except Exception as e:
    print(f"Error loading preprocessor.joblib: {e}")

PREPROCESSORS = {
  "Custom_QNN": load_preprocessor("custom_qnn.joblib"),
  "Hybrid_QNN": load_preprocessor("hybrid_qnn.joblib"),
  "Hybrid_QNN_binary": load_preprocessor("hybrid_qnn_binary.joblib"),
}

def get_regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty and of the same length.")
        
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }

def get_binary_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be of the same length.")
    
    if len(y_true) < 2:
        raise ValueError("At least two prices are required to calculate price direction.")

    # 1. Binarize the true prices
    # Label is 1 if the price increased from the previous step, else 0.
    y_true_bin = [1 if y_true[i] > y_true[i-1] else 0 for i in range(1, len(y_true))]
    
    # 2. Align the predictions
    # Since we lose the first time step calculating the difference, 
    # we must drop the first prediction to maintain index alignment.
    y_pred_aligned = y_pred[1:]
    
    # 3. Guard against single-class batches
    # ROC-AUC will crash if the batch only contains 'Up' or only contains 'Down' movements.
    unique_classes = np.unique(y_true_bin)
    if len(unique_classes) < 2:
        raise ValueError("ROC-AUC requires at least one positive (Up) and one negative (Down) movement in the batch.")

    # 4. Binarize the float probabilities for the accuracy calculation
    # Assuming standard 0.5 threshold for classification
    y_pred_bin = [1 if p >= 0.5 else 0 for p in y_pred_aligned]
    
    # 5. Calculate metrics
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    
    # ROC-AUC uses the raw float probabilities, NOT the binarized predictions
    roc_auc = roc_auc_score(y_true_bin, y_pred_aligned)
    
    return {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc)
    }
# --- Schemas ---
class PricePoint(BaseModel):
    date: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None

class PredictionRequest(BaseModel):
    prices: List[PricePoint]

class BulkModelResponse(BaseModel):
  status: str
  predictions: dict[str, List[float] | List[Dict[str, float]]]
  stats: dict[str, Dict[str, float]]

@app.post("/api/predict", response_model=BulkModelResponse)
def predict(payload: PredictionRequest):
  clean_list = [p for p in payload.prices if p.close is not None]
  n_records = len(clean_list)
  
  SEQUENCE_LENGTH = 5
  if n_records < SEQUENCE_LENGTH:
      raise HTTPException(
          status_code=400, 
          detail=f"Insufficient context. Minimum required length is {SEQUENCE_LENGTH}."
      )
  
  closes = [price.close for price in clean_list]
  ohlcv_list = [
    {
      "open": price.open, 
      "high": price.high, 
      "low": price.low, 
      "close": price.close, 
      "volume": price.volume
    } for price in clean_list
  ]

  custom_preprocessor = PREPROCESSORS.get('Custom_QNN')
  hybrid_preprocessor = PREPROCESSORS.get('Hybrid_QNN')
  hybrid_preprocessor_binary = PREPROCESSORS.get('Hybrid_QNN_binary')

  results = {}
  stats = {}
  for model_name in WEIGHTS.keys():
    target_weights = WEIGHTS.get(model_name)
    match model_name:
      case "lstm":
        predictions_list = lstm.run(target_weights, closes, 3)
        stats[model_name] = get_regression_metrics(closes, predictions_list)
      case "qlstm":
        predictions_list = qlstm.run(target_weights, closes, 3)
        stats[model_name] = get_regression_metrics(closes, predictions_list)
      case "custom_qnn":
        selector, x_scaler, y_scaler, selected_features, candidate_features, lookback, feature_range, sequence_length = custom_preprocessor
        predictions_list = custom.run(target_weights, ohlcv_list, selector, x_scaler, y_scaler, candidate_features, sequence_length) 
        stats[model_name] = get_regression_metrics(closes, predictions_list)
      case "hybrid_qnn1":
        selector, x_scaler, y_scaler, selected_features, candidate_features, lookback, feature_range, sequence_length = hybrid_preprocessor
        predictions_list = hybrid1.run(target_weights, ohlcv_list, selector, x_scaler, y_scaler, candidate_features, 5, 3, 1) 
        stats[model_name] = get_regression_metrics(closes, predictions_list)
      case "hybrid_qnn2":
        selector, x_scaler, y_scaler, selected_features, candidate_features, lookback, feature_range, sequence_length = hybrid_preprocessor
        predictions_list = hybrid2.run(target_weights, ohlcv_list, selector, x_scaler, y_scaler, candidate_features, 5, 3, 1)
        stats[model_name] = get_regression_metrics(closes, predictions_list)
      case "hybrid_qnn1_binary":
        selector, x_scaler, y_scaler, selected_features, candidate_features, lookback, feature_range, sequence_length = hybrid_preprocessor_binary
        predictions_list = hybrid1.run_binary(target_weights, ohlcv_list, selector, x_scaler, candidate_features, 5, 3, 1)
        probabilities_list = [x['probability'] for x in predictions_list]
        stats[model_name] = get_binary_metrics(closes, probabilities_list)
      case "hybrid_qnn2_binary":
        selector, x_scaler, y_scaler, selected_features, candidate_features, lookback, feature_range, sequence_length = hybrid_preprocessor_binary
        predictions_list = hybrid2.run_binary(target_weights, ohlcv_list, selector, x_scaler, candidate_features, 5, 3, 1)
        probabilities_list = [x['probability'] for x in predictions_list]
        stats[model_name] = get_binary_metrics(closes, probabilities_list)

    results[model_name] = predictions_list

  return {
    "status": "success",
    "predictions": results,
    "stats": stats
  }