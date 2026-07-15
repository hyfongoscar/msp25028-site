import numpy as np
import pandas as pd
from typing import List, Dict
import pennylane as qml

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
  
  # Check standard hidden units from weights
  # Usually QLSTM will have input/output weight projections stored in the state dict
  # Modify string lookup names if your custom model layers vary.
  print(weights.keys())
  clayer_in_w = weights["lstm.clayer_in.weight"]   # Shape: (n_qubits, F + H)
  clayer_in_b = weights["lstm.clayer_in.bias"]     # Shape: (n_qubits,)
  
  # Quantum weights shapes are typically: (n_layers, n_qubits)
  vqc_w_f = weights["vqc_f.weights"]
  vqc_w_i = weights["vqc_i.weights"]
  vqc_w_g = weights["vqc_g.weights"]
  vqc_w_o = weights["vqc_o.weights"]
  
  # Output layers to convert expectaton values back to hidden state dimensions
  clayer_out_w = weights["lstm.clayer_out.weight"] # Shape: (4*H, n_qubits)
  clayer_out_b = weights["lstm.clayer_out.bias"]   # Shape: (4*H,)
  
  w_fc = weights["fc.weight"]                 # Shape: (1, H)
  b_fc = weights["fc.bias"]                   # Shape: (1,)

  n_qubits = 4
  n_layers = 16
  @qml.qnode(qml.device("default.qubit", wires=n_qubits), interface="autograd")
  def _quantum_circuit(inputs, weights):
      # Encode incoming features into the qubit states via rotation gates
      for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
      
      # Parameterized quantum circuit layer
      for l in range(n_layers):
        for i in range(n_qubits):
          qml.RY(weights[l, i], wires=i)
        for i in range(n_qubits - 1):
          qml.CNOT(wires=[i, i + 1])
        if n_qubits > 1:
          qml.CNOT(wires=[n_qubits - 1, 0])
              
      return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
  
  qnode = _quantum_circuit
  
  hidden_size = w_fc.shape[1]
  predictions = []
  
  for window in windows:
    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)
    
    for t in range(seq_len):
      x_t = window[t]
      
      # 1. Concatenate current inputs and previous hidden state
      v_concat = np.concatenate([x_t, h])
      
      # 2. Project down to qubits features classically
      v_projected = np.dot(clayer_in_w, v_concat) + clayer_in_b
      
      # Scale values safely to bound angles inside [-pi, pi]
      v_projected = np.arctan(v_projected)
      
      # 3. Process expectation values through 4 separate gate circuits (f, i, g, o)
      q_f = np.array(qnode(v_projected, vqc_w_f))
      q_i = np.array(qnode(v_projected, vqc_w_i))
      q_g = np.array(qnode(v_projected, vqc_w_g))
      q_o = np.array(qnode(v_projected, vqc_w_o))
      
      # Concatenate quantum expectation results
      q_combined = np.concatenate([q_f, q_i, q_g, q_o])
      
      # 4. Map back to gate dimensions classically
      gates = np.dot(clayer_out_w, q_combined) + clayer_out_b
      
      # 5. Apply standard LSTM operations
      i = sigmoid(gates[0 : hidden_size])
      f = sigmoid(gates[hidden_size : 2 * hidden_size])
      g = np.tanh(gates[2 * hidden_size : 3 * hidden_size])
      o = sigmoid(gates[3 * hidden_size : 4 * hidden_size])
      
      c = f * c + i * g
      h = o * np.tanh(c)
        
    out = np.dot(w_fc, h) + b_fc
    predictions.append(out[0])
      
  predictions_arr = np.array(predictions).reshape(-1, 1)
  return predictions_arr.tolist()