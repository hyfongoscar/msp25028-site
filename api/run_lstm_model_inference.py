import numpy as np
import pandas as pd
from typing import List, Dict

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def get_sliding_windows(data: List[float], seq_len: int) -> np.ndarray:
    """Slices a 2D array into a 3D sliding window array of shape (B, T, F)."""
    num_records = len(data)
    num_windows = num_records - seq_len + 1
    windows = []
    for i in range(num_windows):
        windows.append(data[i : i + seq_len])
    return np.array(windows)


def run(
  weights: Dict[str, np.ndarray], 
  close_list: List[float], 
  seq_len: int,
) -> List[float]:
  windows = get_sliding_windows(close_list, seq_len)
  
  # 3. Retrieve weights (Standard PyTorch names exported to numpy)
  w_ih = weights["lstm.weight_ih_l0"]  # Shape: (4*H, F)
  w_hh = weights["lstm.weight_hh_l0"]  # Shape: (4*H, H)
  b_ih = weights["lstm.bias_ih_l0"]    # Shape: (4*H,)
  b_hh = weights["lstm.bias_hh_l0"]    # Shape: (4*H,)
  
  w_fc = weights["fc.weight"]          # Shape: (1, H)
  b_fc = weights["fc.bias"]            # Shape: (1,)
  
  hidden_size = w_hh.shape[1]
  predictions = []
  
  # 4. Iterate through each window batch item
  for window in windows:
    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)
    
    # Step through each sequence step sequentially
    for t in range(seq_len):
      x_t = window[t]  # (F,)
      
      # Stack gate equations
      gates = np.dot(w_ih, x_t) + b_ih + np.dot(w_hh, h) + b_hh
      
      # Split PyTorch's concatenated gate layout: [i, f, g, o]
      i = sigmoid(gates[0 : hidden_size])
      f = sigmoid(gates[hidden_size : 2 * hidden_size])
      g = np.tanh(gates[2 * hidden_size : 3 * hidden_size])
      o = sigmoid(gates[3 * hidden_size : 4 * hidden_size])
      
      c = f * c + i * g
      h = o * np.tanh(c)
        
    # Compute final output projection step
    out = np.dot(w_fc, h) + b_fc
    predictions.append(out[0])
      
  # 5. Inverse-scale results back to actual price domain
  predictions_arr = np.array(predictions).reshape(-1, 1)
  return predictions_arr.tolist()
