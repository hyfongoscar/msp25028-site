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
import api.run_hybrid1_model_inference as hybrid1
import api.run_hybrid2_model_inference as hybrid2

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
  "LSTM": load_model_weights("lstm_weights.npy"),
  "QLSTM": load_model_weights("qlstm_q.npy"),
  "Custom_QNN": load_model_weights("custom_qnn_weights.npy"),
  "Hybrid_QNN1": load_model_weights("hybrid_qnn1_weights.npy"),
  "Hybrid_QNN2": load_model_weights("hybrid_qnn2_weights.npy"),
  "Hybrid_QNN1_binary": load_model_weights("hybrid_qnn1_binary_weights.npy"),
  "Hybrid_QNN2_binary": load_model_weights("hybrid_qnn2_binary_weights.npy"),
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
  "Hybrid_QNN": load_preprocessor("hybrid_qnn.joblib"),
  "Hybrid_QNN_binary": load_preprocessor("hybrid_qnn_binary.joblib"),
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


@app.post("/api/predict", response_model=BulkModelResponse)
def predict(payload: PredictionRequest):
  n_records = len(payload.prices)

  SEQUENCE_LENGTH = 5
  if n_records < SEQUENCE_LENGTH:
      raise HTTPException(
          status_code=400, 
          detail=f"Insufficient context. Minimum required length is {SEQUENCE_LENGTH}."
      )
  
  closes = [price.close for price in payload.prices]
  ohlcv_list = [
    {
      "open": price.open, 
      "high": price.high, 
      "low": price.low, 
      "close": price.close, 
      "volume": price.volume
    } for price in payload.prices
  ]
  selector, x_scaler, y_scaler, selected_features, candidate_features, lookback, feature_range, sequence_length = PREPROCESSORS.get('Hybrid_QNN')

  results = {}
  for model_name in WEIGHTS.keys():
    target_weights = WEIGHTS.get(model_name)
    match model_name:
      case "LSTM":
        # predictions_list = lstm.run(target_weights, closes, 3)
        predictions_list = []
      case "QLSTM":
        predictions_list = qlstm.run(target_weights, closes, 3)
      case "Custom_QNN":
        predictions_list = []
        # predictions_list = qlstm.run(target_weights, closes, SEQUENCE_LENGTH)
      case "Hybrid_QNN1":
        predictions_list = hybrid1.run(target_weights, ohlcv_list, selector, x_scaler, y_scaler, candidate_features, sequence_length, 3, 1) 
      case "Hybrid_QNN2":
        predictions_list = hybrid2.run(target_weights, ohlcv_list, selector, x_scaler, y_scaler, candidate_features, sequence_length, 3, 1)
      case "Hybrid_QNN1_binary":
        predictions_list = hybrid1.run_binary(target_weights, ohlcv_list, selector, x_scaler, candidate_features, sequence_length, 3, 1)
      case "Hybrid_QNN2_binary":
        predictions_list = hybrid2.run_binary(target_weights, ohlcv_list, selector, x_scaler, candidate_features, sequence_length, 3, 1)

    results[model_name] = predictions_list

  return {
    "status": "success",
    "predictions": results
  }