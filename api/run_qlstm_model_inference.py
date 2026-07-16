import numpy as np
import pandas as pd
from typing import List, Dict, Any
import pennylane as qml

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def get_sliding_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    windows = []
    for i in range(len(data)):
        if i < seq_len:
          padding = np.zeros(seq_len - i - 1).reshape(-1, data.shape[1])
          windows.append(np.concatenate((padding, data[:i+1]), axis=0))
        else:
          windows.append(data[i - seq_len + 1: i + 1])
    return np.array(windows)

def normalize(a, min_a=None, max_a=None):
    min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def undo_normalize(a, min_a, max_a):
    return a * (max_a - min_a + 0.0001) + min_a

def run(
  master_weights: dict, 
  close_prices: List[float],
  seq_len: int
) -> list:
  clayer_in_w = np.array(master_weights["lstm.clayer_in.weight"], dtype=float)   # Shape: (4, 17)
  clayer_in_b = np.array(master_weights["lstm.clayer_in.bias"], dtype=float)     # Shape: (4,)
  
  clayer_out_w = np.array(master_weights["lstm.clayer_out.weight"], dtype=float) # Shape: (16, 4)
  clayer_out_b = np.array(master_weights["lstm.clayer_out.bias"], dtype=float)   # Shape: (16,)
  
  linear_w = np.array(master_weights["linear.weight"], dtype=float)             # Shape: (1, 16)
  linear_b = np.array(master_weights["linear.bias"], dtype=float)
  
  hidden_size = 16  # Confirmed by linear_w shape
  seq_len = 3      # Hardcoded sequence length from training
  
  w_forget = master_weights["lstm.VQC.forget.weights"]
  w_input = master_weights["lstm.VQC.input.weights"]
  w_update = master_weights["lstm.VQC.q_update.weights"]
  w_output = master_weights["lstm.VQC.output.weights"]

  w_forget = np.array(w_forget, dtype=float)
  w_input = np.array(w_input, dtype=float)
  w_update = np.array(w_update, dtype=float)
  w_output = np.array(w_output, dtype=float)

  n_qubits = 4
  n_layers = 1
        
  @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="autograd")
  def _quantum_circuit(inputs, weights):
    for i in range(n_qubits):
      qml.Hadamard(wires=i) 
      qml.RY(inputs[i], wires=i)
      qml.RZ(inputs[i]**2, wires=i)
  
    for l in range(n_layers):
      for i in range(1, 3):
        for j in range(n_qubits):
          if j + i < n_qubits:
            qml.CNOT(wires=[j, j + i])
          else:
            qml.CNOT(wires=[j, j + i - n_qubits])

      for i in range(n_qubits):
        qml.RX(weights[l, 0, i], wires=i) 
        qml.RY(weights[l, 1, i], wires=i)
        qml.RZ(weights[l, 2, i], wires=i)
        
    # 3. Measurement
    # Returns the expectation value in the Z basis (Outputs a classical float scalar)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
      
  qnode = _quantum_circuit

  close_prices_arr = np.array(close_prices).reshape(-1, 1)
  close_prices_arr, min_close, max_close = normalize(close_prices_arr)

  windows = get_sliding_windows(close_prices_arr, seq_len)
  
  predictions = []
  
  # 4. Core Recurrent Inference Loop
  for window in windows:
    h_t = np.zeros(hidden_size, dtype=float)
    c_t = np.zeros(hidden_size, dtype=float)
    
    for t in range(seq_len):
      x_t = window[t] 
      
      # Defense 1: Ensure concatenated vector drops any object flags
      v_t = np.concatenate([h_t, x_t]).astype(float)
      
      # Defense 2: Strip object wrapper status before running arctan
      y_t = np.dot(clayer_in_w, v_t) + clayer_in_b
      y_t = y_t.astype(float) 
      
      # Execute the 4 quantum node gates
      q_f = np.array(qnode(y_t, w_forget)).astype(float)
      q_i = np.array(qnode(y_t, w_input)).astype(float)
      q_g = np.array(qnode(y_t, w_update)).astype(float)
      q_o = np.array(qnode(y_t, w_output)).astype(float)
      
      # Defense 3: Enforce hard float64 conversion for the classical projections
      f_t = sigmoid((np.dot(clayer_out_w, q_f) + clayer_out_b).astype(float))
      i_t = sigmoid((np.dot(clayer_out_w, q_i) + clayer_out_b).astype(float))
      g_t = np.tanh((np.dot(clayer_out_w, q_g) + clayer_out_b).astype(float))
      o_t = sigmoid((np.dot(clayer_out_w, q_o) + clayer_out_b).astype(float))
      
      # Defense 4: Sanitize recurrent states before they loop back to t+1
      c_t = (f_t * c_t + i_t * g_t).astype(float)
      h_t = (o_t * np.tanh(c_t)).astype(float)
        
    # Final regression layer pass
    out = np.dot(linear_w, h_t) + linear_b
    predictions.append(undo_normalize(out[0], min_close, max_close))
      
  if not predictions:
    return []
      
  # 5. Inverse-scale predictions back to standard price values
  predictions_arr = np.array(predictions).reshape(-1)
  return predictions_arr.tolist()