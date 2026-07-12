import os
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Q-Net Finance Selective Model Backend")

SEQUENCE_LENGTH = 5
CURRENT_DIR = Path(__file__).parent

def load_model_weights(model_file_name: str) -> dict:
  file_path = CURRENT_DIR / "weights" / model_file_name
  if not file_path.exists():
    print(f"Warning: Artifact {model_file_name} not found.")
    return None
  return np.load(file_path, allow_pickle=True).item()
  # return torch.load(file_path, map_location=torch.device('cpu'))

# Load the custom state payloads on server spin-up
WEIGHTS = {
  "QLSTM": load_model_weights("qlstm_weights.npy"),
  "Custom_QNN": load_model_weights("custom_qnn_weights.npy"),
  "Hybrid_QNN1": load_model_weights("hybrid_qnn1_weights.npy"),
}

# --- REFACTORED INFERENCE ENGINE FOR BULK FORECASTING ---
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
  
  # 1. Normalize the entire historical stream using your training formula
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

  # 2. Replicate the DataLoader loop by sliding across the index timeline
  for i in range(len(scaled_prices)):
    # Construct the padded or sliding sequence window exactly like SequenceDataset
    if i >= SEQUENCE_LENGTH - 1:
      window = scaled_prices[i - SEQUENCE_LENGTH + 1 : i + 1]
    else:
      # Padding replication using the first element if the array is short at index 0
      padding = np.repeat(scaled_prices[0], SEQUENCE_LENGTH - i - 1)
      window = np.concatenate([padding, scaled_prices[0 : i + 1]])
        
    # 3. Compute Forward Pass Matrix evaluation per step, use numpy instead of torch
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

class BulkModelResponse(BaseModel):
  status: str
  predictions: dict[str, List[float]]

@app.get("/api/predict", response_model=BulkModelResponse)
def predict(prices: str):
  try:
    # 2. Strip the URL-encoded brackets '[' and ']' automatically
    cleaned_prices = prices.strip().lstrip("[").rstrip("]")
    
    # 3. Parse safely into floats
    price_list = [float(p.strip()) for p in cleaned_prices.split(",") if p.strip()]
  except ValueError:
    raise HTTPException(status_code=400, detail="Invalid numerical data in prices parameter.")
      
  if len(price_list) < SEQUENCE_LENGTH:
    raise HTTPException(status_code=400, detail="Insufficient price context length.")

  results = {}
  for model_name in WEIGHTS.keys():
    target_weights = WEIGHTS.get(model_name)
    predictions_list = run_bulk_model_inference(target_weights, price_list)
    results[model_name] = predictions_list

  return {
    "status": "success",
    "predictions": results
  }