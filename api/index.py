import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title="Q-Net Finance Selective Model Backend")

SEQUENCE_LENGTH = 5
CURRENT_DIR = Path(__file__).parent

def load_model_weights(model_file_name: str) -> dict:
  file_path = CURRENT_DIR / "weights" / model_file_name
  if not file_path.exists():
    print(f"Warning: Artifact {model_file_name} not found.")
    return None
  return np.load(file_path, allow_pickle=True).item()

# Load the custom state payloads on server spin-up
WEIGHTS = {
  "QLSTM": load_model_weights("qlstm_weights.npy"),
  "Custom_QNN": load_model_weights("custom_qnn_weights.npy"),
  "Hybrid_QNN1": load_model_weights("hybrid_qnn1_weights.npy"),
  "Hybrid_QNN2": load_model_weights("hybrid_qnn2_weights.npy"),
}

def load_preprocessor(preprocessor_file_name: str):
  try:
    # 1. Load the joblib file
    JOBLIB_PATH = CURRENT_DIR / "preprocessors" / preprocessor_file_name
    preprocessor = joblib.load(JOBLIB_PATH)
    
    # 2. Extract components dynamically based on how your teammate structured it
    # Case A: If it's saved as a dictionary:
    if isinstance(preprocessor, dict):
      selector = preprocessor.get("selector")
      x_scaler = preprocessor.get("x_scaler")
      y_scaler = preprocessor.get("y_scaler")
      candidate_features = preprocessor.get("candidate_features")
      # Fallback to your default config value if lookback/sequence_length isn't in joblib
      sequence_length = preprocessor.get("lookback", 10) 
        
    # Case B: If it's a custom wrapper class object (e.g., PreparedData or Preprocessor):
    else:
      selector = getattr(preprocessor, "selector", None)
      x_scaler = getattr(preprocessor, "x_scaler", None)
      y_scaler = getattr(preprocessor, "y_scaler", None)
      candidate_features = getattr(preprocessor, "candidate_features", None)
      sequence_length = getattr(preprocessor, "lookback", 10)

    return selector, x_scaler, y_scaler, candidate_features, sequence_length
  except Exception as e:
    print(f"Error loading preprocessor.joblib: {e}")

PREPROCESSORS = {
  "Hybrid_QNN1": load_preprocessor("hybrid_qnn1.joblib"),
  "Hybrid_QNN2": load_preprocessor("hybrid_qnn2.joblib"),
}

def run_bulk_model_inference(state_dict: dict, raw_prices: list) -> list:
  """
  Slided a sequence window across the entire historical price list,
  generating a matching list of predictions for historical visualization.
  """
  if not raw_prices or len(raw_prices) < SEQUENCE_LENGTH:
      return []
      
  prices_arr = np.array(raw_prices, dtype=np.float32)
  min_val = np.min(prices_arr)
  max_val = np.max(prices_arr)
  
  scaled_prices = (prices_arr - min_val) / (max_val - min_val + 0.0001)
  
  predictions_scaled = []
  
  weight_matrix = None
  try:
    weight_key = [k for k in state_dict.keys() if 'weight' in k][0]
    weight_matrix = state_dict[weight_key]
  except Exception:
    print("Warning: Model artifacts not found.")
      
  if weight_matrix is None:
    raise HTTPException(status_code=500, detail="Model artifacts not found.")

  for i in range(len(scaled_prices)):
    # Construct the padded or sliding sequence window exactly like SequenceDataset
    if i >= SEQUENCE_LENGTH - 1:
      window = scaled_prices[i - SEQUENCE_LENGTH + 1 : i + 1]
    else:
      # Padding replication using the first element if the array is short at index 0
      padding = np.repeat(scaled_prices[0], SEQUENCE_LENGTH - i - 1)
      window = np.concatenate([padding, scaled_prices[0 : i + 1]])
        
    try:
      flat_x = window.flatten()
      if weight_matrix.ndim == 1:
        target_dim = weight_matrix.shape[0]
        weight_vec = weight_matrix
      else:
        target_dim = weight_matrix.shape[1]
        weight_vec = weight_matrix[0].flatten()
      
      if len(flat_x) > target_dim:
        flat_x = flat_x[:target_dim]
      elif len(flat_x) < target_dim:
        flat_x = np.concatenate([flat_x, np.zeros(target_dim - len(flat_x))])
      
      weight_vec = weight_vec[:len(flat_x)]
      step_pred = float(np.dot(weight_vec, flat_x))
    except Exception as e:
      print('Error during inference:', e)
      step_pred = float(window[-1])
        
    predictions_scaled.append(step_pred)
      
  # 4. Denormalize the entire output array back to raw dollar values
  predictions_unscaled = [float(p * (max_val - min_val) + min_val) for p in predictions_scaled]
  return predictions_unscaled

def canonicalize_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize case variants such as `Close` and `Volume` to lower case."""

    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [str(column[0]) for column in frame.columns]

    renamed: Dict[Any, str] = {}
    for column in frame.columns:
        normalized = str(column).strip().lower()
        if normalized in {"open", "high", "low", "close", "volume"}:
            renamed[column] = normalized
    output = frame.rename(columns=renamed).copy()

    required = {"open", "high", "low", "close"}
    missing = required.difference(output.columns)
    if missing:
        raise ValueError(
            "The input parquet file must provide OHLC columns. "
            f"Missing after normalization: {sorted(missing)}"
        )

    if not output.index.is_monotonic_increasing:
        output = output.sort_index()
    return output

def add_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the RSI, MACD, and ADX indicators used by the notebook pipeline."""

    output = frame.copy()
    delta = output["close"].diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = -delta.where(delta < 0.0, 0.0)
    average_gain = gain.rolling(14).mean()
    average_loss = loss.rolling(14).mean()
    rs = average_gain / (average_loss + 1e-12)
    output["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    ema_12 = output["close"].ewm(span=12, adjust=False).mean()
    ema_26 = output["close"].ewm(span=26, adjust=False).mean()
    output["macd"] = ema_12 - ema_26
    output["macd_signal"] = output["macd"].ewm(span=9, adjust=False).mean()

    high_low = output["high"] - output["low"]
    high_close = (output["high"] - output["close"].shift(1)).abs()
    low_close = (output["low"] - output["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    up_move = output["high"].diff()
    down_move = -output["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    atr_14 = true_range.rolling(14).mean()
    plus_di = 100.0 * (
        pd.Series(plus_dm, index=output.index).rolling(14).mean() / (atr_14 + 1e-12)
    )
    minus_di = 100.0 * (
        pd.Series(minus_dm, index=output.index).rolling(14).mean() / (atr_14 + 1e-12)
    )
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12))
    output["adx_14"] = dx.rolling(14).mean()

    # Preserve the original timeline length so the model can score every input row.
    # Early indicator values are undefined, so fill them conservatively and keep the
    # sequence aligned with the input OHLCV history.
    return output.ffill().fillna(0.0)

def run_hybrid_bulk_model_inference(
    state_dict: dict, 
    raw_ohlcv_list: list,
    selector,
    x_scaler,
    y_scaler,
    candidate_features,
    sequence_length
) -> list:
  """
  Slides a multi-feature sequence window across historical OHLCV data,
  matching training indicators, SelectKBest features, and scaling constraints.
  """
  if not raw_ohlcv_list or len(raw_ohlcv_list) < sequence_length:
    return []
      
  df = pd.DataFrame(raw_ohlcv_list)
  df = canonicalize_ohlcv_columns(df)
  df = add_technical_indicators(df)
  
  if len(df) == 0:
    return []

  # Extract raw target arrays and feature matrices matching training candidates
  # close_values = df["close"].to_numpy(dtype=np.float64)
  features_df = df[candidate_features]
    
  # 2. Apply the pre-fitted SelectKBest transformation and MinMaxScaler 
  # This maps your Open, High, Low, and engineered indicators down to the exact k features
  selected_features_matrix = selector.transform(features_df)
  X_scaled = x_scaler.transform(selected_features_matrix)
  
  # Extract model weights
  weight_matrix = None
  try:
    weight_key = [k for k in state_dict.keys() if 'weight' in k][0]
    weight_matrix = state_dict[weight_key]
  except Exception:
    raise HTTPException(status_code=500, detail="Model weights artifact not found.")

  predictions_scaled = []
  row_count = len(X_scaled)

  # 3. Slide across the timeline
  for i in range(row_count):
    if i >= sequence_length - 1:
      window = X_scaled[i - sequence_length + 1 : i + 1]
    else:
      # Replicate padding: repeat the feature row of the very first index
      padding_count = sequence_length - i - 1
      padding = np.repeat(X_scaled[0:1], padding_count, axis=0)
      window = np.concatenate([padding, X_scaled[0 : i + 1]], axis=0)
          
    # 4. Flatten the 2D window (Shape: sequence_length * k_features)
    try:
      flat_x = window.flatten()
      
      if weight_matrix.ndim == 1:
        target_dim = weight_matrix.shape[0]
        weight_vec = weight_matrix
      else:
        target_dim = weight_matrix.shape[1]
        weight_vec = weight_matrix[0].flatten()
        
      # Re-verify dimensions match weight requirements
      if len(flat_x) > target_dim:
        flat_x = flat_x[:target_dim]
      elif len(flat_x) < target_dim:
        flat_x = np.concatenate([flat_x, np.zeros(target_dim - len(flat_x))])
      
      weight_vec = weight_vec[:len(flat_x)]
      step_pred = float(np.dot(weight_vec, flat_x))
          
    except Exception as e:
      print('Error during inference:', e)
      # Fallback strategy: match normalized scale of the previous step close price
      step_pred = float(X_scaled[max(0, i-1), 0])
          
    predictions_scaled.append(step_pred)
      
  # 5. Denormalize your output using the actual fitted y_scaler from training
  predictions_scaled_arr = np.array(predictions_scaled, dtype=np.float64).reshape(-1, 1)
  predictions_unscaled = y_scaler.inverse_transform(predictions_scaled_arr).flatten().tolist()
  
  return [float(p) for p in predictions_unscaled]

class BulkModelResponse(BaseModel):
  status: str
  predictions: dict[str, List[float]]

@app.get("/api/predict", response_model=BulkModelResponse)
def predict(opens: str, highs: str, lows: str, closes: str, volumes: str, adjCloses: str):
  try:
    cleaned_opens = opens.strip().lstrip("[").rstrip("]")
    cleaned_highs = highs.strip().lstrip("[").rstrip("]")
    cleaned_lows = lows.strip().lstrip("[").rstrip("]")
    cleaned_closes = closes.strip().lstrip("[").rstrip("]")
    cleaned_volumes = volumes.strip().lstrip("[").rstrip("]")
    cleaned_adjCloses = adjCloses.strip().lstrip("[").rstrip("]")
    
    # 3. Parse safely into floats
    close_list = [float(p.strip()) for p in cleaned_closes.split(",") if p.strip()]
  except ValueError:
    raise HTTPException(status_code=400, detail="Invalid numerical data in prices parameter.")
      
  if len(close_list) < SEQUENCE_LENGTH:
    raise HTTPException(status_code=400, detail="Insufficient price context length.")

  results = {}
  for model_name in WEIGHTS.keys():
    target_weights = WEIGHTS.get(model_name)
    if model_name == "QLSTM" or model_name == "Custom_QNN":
      predictions_list = run_bulk_model_inference(target_weights, close_list)
      results[model_name] = predictions_list
    else:
      ohlcv_list = [
        {
          "open": float(open),
          "high": float(high),
          "low": float(low),
          "close": float(close),
          "volume": float(volume),
          "adjClose": float(adjClose)
        } for open, high, low, close, volume, adjClose in zip(cleaned_opens.split(","), cleaned_highs.split(","), cleaned_lows.split(","), cleaned_closes.split(","), cleaned_volumes.split(","), cleaned_adjCloses.split(","))
      ]
      selector, x_scaler, y_scaler, candidate_features, sequence_length = PREPROCESSORS.get(model_name)
      predictions_list = run_hybrid_bulk_model_inference(target_weights, ohlcv_list, selector, x_scaler, y_scaler, candidate_features, sequence_length)
      results[model_name] = predictions_list

  return {
    "status": "success",
    "predictions": results
  }