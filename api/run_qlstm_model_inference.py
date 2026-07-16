import numpy as np
import pandas as pd
from typing import List, Dict, Any
import pennylane as qml

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def get_sliding_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    num_records = len(data)
    num_windows = num_records - seq_len + 1
    windows = []
    for i in range(num_windows):
        windows.append(data[i : i + seq_len])
    return np.array(windows)

def run(
  master_weights: dict, 
  close_prices: List[float],
  seq_len: int
) -> list:
  clayer_in_w = np.array(master_weights["state_dict.lstm.clayer_in.weight"], dtype=float)   # Shape: (4, 17)
  clayer_in_b = np.array(master_weights["state_dict.lstm.clayer_in.bias"], dtype=float)     # Shape: (4,)
  
  clayer_out_w = np.array(master_weights["state_dict.lstm.clayer_out.weight"], dtype=float) # Shape: (16, 4)
  clayer_out_b = np.array(master_weights["state_dict.lstm.clayer_out.bias"], dtype=float)   # Shape: (16,)
  
  linear_w = np.array(master_weights["state_dict.linear.weight"], dtype=float)             # Shape: (1, 16)
  linear_b = np.array(master_weights["state_dict.linear.bias"], dtype=float)
  
  hidden_size = 16  # Confirmed by linear_w shape
  seq_len = 3      # Hardcoded sequence length from training
  
  # 2. Extract the 4 sets of independent quantum gate weights
  w_forget = master_weights.get("model['lstm'].qlayer_forget._tape._ops[0].data[0]")
  w_input = master_weights.get("model['lstm'].qlayer_input._tape._ops[0].data[0]")
  w_update = master_weights.get("model['lstm'].qlayer_update._tape._ops[0].data[0]")
  w_output = master_weights.get("model['lstm'].qlayer_output._tape._ops[0].data[0]")
  
  if w_forget is None:
    w_forget = master_weights["model.lstm.qlayer_forget._tape._ops[0].data[0]"]
    w_input = master_weights["model.lstm.qlayer_input._tape._ops[0].data[0]"]
    w_update = master_weights["model.lstm.qlayer_update._tape._ops[0].data[0]"]
    w_output = master_weights["model.lstm.qlayer_output._tape._ops[0].data[0]"]

  w_forget = np.array(w_forget, dtype=float)
  w_input = np.array(w_input, dtype=float)
  w_update = np.array(w_update, dtype=float)
  w_output = np.array(w_output, dtype=float)

  n_qubits = 4
  n_layers = 1
        
  @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="autograd")
  def _quantum_circuit(inputs, weights):
    # Angle Encoding
    for i in range(n_qubits):
      qml.RY(inputs[i], wires=i)
    # Variational Layers
    for l in range(n_layers):
      for i in range(n_qubits):
        qml.RY(weights[l, i], wires=i)
      for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
      if n_qubits > 1:
        qml.CNOT(wires=[n_qubits - 1, 0]) 
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
      
  qnode = _quantum_circuit

  close_prices_arr = np.array(close_prices).reshape(-1, 1)
  windows = get_sliding_windows(close_prices_arr, seq_len)
  
  predictions = []
  
  # 4. Core Recurrent Inference Loop
  for window in windows:
    h = np.zeros(hidden_size, dtype=float)
    c = np.zeros(hidden_size, dtype=float)
    
    for t in range(seq_len):
      x_t = window[t] 
      
      # Defense 1: Ensure concatenated vector drops any object flags
      v_concat = np.concatenate([x_t, h]).astype(float)
      
      # Defense 2: Strip object wrapper status before running arctan
      v_projected = np.dot(clayer_in_w, v_concat) + clayer_in_b
      v_projected = v_projected.astype(float) 
      v_projected = np.arctan(v_projected) 
      
      # Execute the 4 quantum node gates
      q_f = np.array(qnode(v_projected, w_forget)).astype(float)
      q_i = np.array(qnode(v_projected, w_input)).astype(float)
      q_g = np.array(qnode(v_projected, w_update)).astype(float)
      q_o = np.array(qnode(v_projected, w_output)).astype(float)
      
      # Defense 3: Enforce hard float64 conversion for the classical projections
      f = sigmoid((np.dot(clayer_out_w, q_f) + clayer_out_b).astype(float))
      i = sigmoid((np.dot(clayer_out_w, q_i) + clayer_out_b).astype(float))
      g = np.tanh((np.dot(clayer_out_w, q_g) + clayer_out_b).astype(float))
      o = sigmoid((np.dot(clayer_out_w, q_o) + clayer_out_b).astype(float))
      
      # Defense 4: Sanitize recurrent states before they loop back to t+1
      c = (f * c + i * g).astype(float)
      h = (o * np.tanh(c)).astype(float)
        
    # Final regression layer pass
    out = np.dot(linear_w, h) + linear_b
    predictions.append(float(out[0]))
      
  if not predictions:
    return []
      
  # 5. Inverse-scale predictions back to standard price values
  predictions_arr = np.array(predictions).reshape(-1)
  return predictions_arr.tolist()