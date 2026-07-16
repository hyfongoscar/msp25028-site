import numpy as np
import pandas as pd
from typing import List, Dict

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def normalize(a, min_a=None, max_a=None):
    min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def undo_normalize(a, min_a, max_a):
    return a * (max_a - min_a + 0.0001) + min_a

def get_sliding_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    windows = []
    for i in range(len(data)):
        if i < seq_len:
          padding = np.zeros(seq_len - i - 1)
          windows.append(np.concatenate((padding, data[:i+1]), axis=0))
        else:
          windows.append(data[i - seq_len + 1: i + 1])
    return np.array(windows)

def run(
  weights: Dict[str, np.ndarray], 
  close_list: List[float], 
  seq_len: int
) -> List[float]:
  close_prices_arr, min_close, max_close = normalize(close_list)
  windows = get_sliding_windows(close_prices_arr, seq_len)
  
  # 1. Retrieve weights (Standard PyTorch names exported to numpy)
  w_ih = weights["lstm.weight_ih_l0"]  # Shape: (4*H, F)
  w_hh = weights["lstm.weight_hh_l0"]  # Shape: (4*H, H)
  b_ih = weights["lstm.bias_ih_l0"]    # Shape: (4*H,)
  b_hh = weights["lstm.bias_hh_l0"]    # Shape: (4*H,)
  
  w_fc = weights["linear.weight"]      # Shape: (1, H)
  b_fc = weights["linear.bias"]        # Shape: (1,)
  
  hidden_size = w_hh.shape[1]
  predictions = []
  
  # 2. Iterate through each window batch item
  for window in windows:
    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)
    
    # Step through each sequence step sequentially
    for t in range(seq_len):
      # CRITICAL FIX: Wrap scalar float into a 1D array of shape (1,) 
      # so np.dot((4*H, 1), (1,)) results in a 1D array of shape (4*H,)
      x_t = np.array([window[t]]) 
      
      # Stack gate equations
      gates = np.dot(w_ih, x_t) + b_ih + np.dot(w_hh, h) + b_hh
      
      # Split PyTorch's concatenated gate layout: [i, f, g, o]
      i = sigmoid(gates[0 : hidden_size])
      f = sigmoid(gates[hidden_size : 2 * hidden_size])
      g = np.tanh(gates[2 * hidden_size : 3 * hidden_size])
      o = sigmoid(gates[3 * hidden_size : 4 * hidden_size])
      
      # Update cell and hidden states
      c = f * c + i * g
      h = o * np.tanh(c)
        
    # Compute final output projection step for the current window
    out = np.dot(w_fc, h) + b_fc
    predictions.append(undo_normalize(out[0], min_close, max_close))
      
  # 3. Format results into a 2D list of shape (B, 1)
  predictions_arr = np.array(predictions).reshape(-1)
  return predictions_arr.tolist()