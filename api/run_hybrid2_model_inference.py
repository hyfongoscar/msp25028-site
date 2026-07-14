import numpy as np
import pandas as pd
import pennylane as qml
from fastapi import HTTPException

from api.run_hybrid1_model_inference import canonicalize_ohlcv_columns, add_technical_indicators

WINDOW_DAYS = 10

def forward_numpy_lstm(X: np.ndarray, w: dict) -> np.ndarray:
  batch_size, seq_len, _ = X.shape
  
  # Extract structural LSTM parameters
  w_ih = w["lstm.weight_ih_l0"]
  w_hh = w["lstm.weight_hh_l0"]
  b_ih = w["lstm.bias_ih_l0"]
  b_hh = w["lstm.bias_hh_l0"]
  
  hidden_size = w_hh.shape[1]
  last_hidden_states = []
  
  for b in range(batch_size):
    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)
    
    # Recurrent calculation loop
    for t in range(seq_len):
      x_t = X[b, t, :]
      gates = np.dot(w_ih, x_t) + b_ih + np.dot(w_hh, h) + b_hh
      i, f, g, o = np.split(gates, 4)
      
      i = 1.0 / (1.0 + np.exp(-i))
      f = 1.0 / (1.0 + np.exp(-f))
      g = np.tanh(g)
      o = 1.0 / (1.0 + np.exp(-o))
      
      c = f * c + i * g
      h = o * np.tanh(c)
        
    last_hidden_states.append(h)
      
  return np.array(last_hidden_states, dtype=np.float32)

def run(
  weights_dict: dict,
  ohlcv_list: list, 
  selector, 
  x_scaler, 
  y_scaler,
  candidate_features: list,
  sequence_length: int,
  n_qubits: int,
  q_layers: int
) -> list:
  df = pd.DataFrame(ohlcv_list)
  df = canonicalize_ohlcv_columns(df)  
  df = add_technical_indicators(df).dropna()

  features_df = df[candidate_features].copy() if candidate_features else df.copy()

  for column in candidate_features:
    values = features_df[column].to_numpy(dtype=np.float64)
    features_df[column] = pd.Series(values).rolling(window=WINDOW_DAYS, min_periods=1).mean().to_numpy(dtype=np.float64)

  selected_features_matrix = selector.transform(features_df)
  X_scaled = x_scaler.transform(selected_features_matrix)
  row_count = len(X_scaled)

  sequences = []
  for i in range(row_count):
    if i >= sequence_length - 1:
      window = X_scaled[i - sequence_length + 1 : i + 1]
    else:
      padding_count = sequence_length - i - 1
      padding = np.repeat(X_scaled[0:1], padding_count, axis=0)
      window = np.concatenate([padding, X_scaled[0 : i + 1]], axis=0)
    sequences.append(window)
      
  X_array = np.array(sequences, dtype=np.float32) 

  batch_size, seq_len, n_feats = X_array.shape
  hidden_size = 16
  
  # 1. Classical Stream: Extract matching PyTorch weights
  w_ih = weights_dict["lstm.weight_ih_l0"]
  w_hh = weights_dict["lstm.weight_hh_l0"]
  b_ih = weights_dict["lstm.bias_ih_l0"]
  b_hh = weights_dict["lstm.bias_hh_l0"]
  
  classical_features = []
  for b in range(batch_size):
    h = np.zeros(hidden_size, dtype=np.float32)
    c = np.zeros(hidden_size, dtype=np.float32)
    for t in range(seq_len):
      x_t = X_array[b, t, :]
      gates = np.dot(w_ih, x_t) + b_ih + np.dot(w_hh, h) + b_hh
      i = 1.0 / (1.0 + np.exp(-gates[0:hidden_size]))
      f = 1.0 / (1.0 + np.exp(-gates[hidden_size:2*hidden_size]))
      g = np.tanh(gates[2*hidden_size:3*hidden_size])
      o = 1.0 / (1.0 + np.exp(-gates[3*hidden_size:4*hidden_size]))
      c = f * c + i * g
      h = o * np.tanh(c)
    classical_features.append(h)
  classical_features = np.array(classical_features, dtype=np.float32)

  flattened_inputs = X_array.reshape(batch_size, -1) 
  q_proj_w = weights_dict["quantum_projection.weight"]
  q_proj_b = weights_dict["quantum_projection.bias"]
  
  quantum_inputs = np.tanh(np.dot(flattened_inputs, q_proj_w.T) + q_proj_b)

  # 3. PennyLane Loop
  dev = qml.device("default.qubit", wires=n_qubits)
  @qml.qnode(dev, interface=None)
  def pure_numpy_qnn_circuit(features, weights):
    clipped = qml.math.clip(features, -0.999999, 0.999999)
    for wire in range(n_qubits):
      qml.RY(qml.math.arcsin(clipped[wire]), wires=wire)
      qml.RZ(qml.math.arccos(clipped[wire]), wires=wire)

    for layer in range(q_layers):
      for wire in range(n_qubits):
        qml.RY(weights[layer, wire, 0], wires=wire)
        qml.RZ(weights[layer, wire, 1], wires=wire)
      for wire in range(n_qubits - 1):
        qml.CNOT(wires=[wire, wire + 1])
        qml.CZ(wires=[wire, wire + 1])

    return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

  q_weights = weights_dict["q_weights"]
  quantum_features = np.array([pure_numpy_qnn_circuit(sample, q_weights) for sample in quantum_inputs], dtype=np.float32)

  # 4. Sequential Prediction Head Mapping
  fused_features = np.concatenate([classical_features, quantum_features], axis=1)
  
  fc1_w = weights_dict["output_head.0.weight"]
  fc1_b = weights_dict["output_head.0.bias"]
  fc2_w = weights_dict["output_head.2.weight"]
  fc2_b = weights_dict["output_head.2.bias"]

  # Manual ReLU implementation matching training sequential layer
  layer1_out = np.maximum(0, np.dot(fused_features, fc1_w.T) + fc1_b)
  predictions_scaled = np.dot(layer1_out, fc2_w.T) + fc2_b

  return y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten().tolist()

def run_binary(
  weights_dict: dict,        
  ohlcv_list: list, 
  selector, 
  x_scaler, 
  candidate_features: list,
  sequence_length: int,
  n_qubits: int,
  q_layers: int
) -> list:
  if not ohlcv_list or len(ohlcv_list) < sequence_length:
    return []
      
  df = pd.DataFrame(ohlcv_list)
  df = canonicalize_ohlcv_columns(df)
  df = add_technical_indicators(df)
  
  if len(df) == 0:
    return []

  features_df = df[candidate_features].copy() if candidate_features else df.copy()
  
  for column in features_df.columns:
    values = features_df[column].to_numpy(dtype=np.float64)
    features_df[column] = pd.Series(values).rolling(window=WINDOW_DAYS, min_periods=1).mean().to_numpy(dtype=np.float64)

  selected_features_matrix = selector.transform(features_df)
  X_scaled = x_scaler.transform(selected_features_matrix)
  row_count = len(X_scaled)

  sequences = []
  for i in range(row_count):
    if i >= sequence_length - 1:
      window = X_scaled[i - sequence_length + 1 : i + 1]
    else:
      padding_count = sequence_length - i - 1
      padding = np.repeat(X_scaled[0:1], padding_count, axis=0)
      window = np.concatenate([padding, X_scaled[0 : i + 1]], axis=0)
    sequences.append(window)
      
  X_array = np.array(sequences, dtype=np.float32) 
  batch_size, seq_len, n_feats = X_array.shape
  hidden_size = 16

  # 1. Evaluate classical hidden states path
  w_ih = weights_dict["lstm.weight_ih_l0"]
  w_hh = weights_dict["lstm.weight_hh_l0"]
  b_ih = weights_dict["lstm.bias_ih_l0"]
  b_hh = weights_dict["lstm.bias_hh_l0"]
  
  classical_features = []
  for b in range(batch_size):
    h = np.zeros(hidden_size, dtype=np.float32)
    c = np.zeros(hidden_size, dtype=np.float32)
    for t in range(seq_len):
      x_t = X_array[b, t, :]
      gates = np.dot(w_ih, x_t) + b_ih + np.dot(w_hh, h) + b_hh
      
      i = 1.0 / (1.0 + np.exp(-gates[0:hidden_size]))
      f = 1.0 / (1.0 + np.exp(-gates[hidden_size:2*hidden_size]))
      g = np.tanh(gates[2*hidden_size:3*hidden_size])
      o = 1.0 / (1.0 + np.exp(-gates[3*hidden_size:4*hidden_size]))
      
      c = f * c + i * g
      h = o * np.tanh(c)
    classical_features.append(h)
  classical_features = np.array(classical_features, dtype=np.float32)

  # 2. Concurrently evaluate quantum mapping path via standard C-contiguous flattening
  flattened_inputs = X_array.reshape(batch_size, -1) 
  q_proj_w = weights_dict["quantum_projection.weight"]
  q_proj_b = weights_dict["quantum_projection.bias"]
  quantum_inputs = np.tanh(np.dot(flattened_inputs, q_proj_w.T) + q_proj_b)

  # 3. PennyLane execution circuit node loop
  dev = qml.device("default.qubit", wires=n_qubits)
  
  @qml.qnode(dev, interface=None)
  def pure_numpy_qnn_circuit(features, weights):
    clipped = qml.math.clip(features, -0.999999, 0.999999)
    for wire in range(n_qubits):
      qml.RY(qml.math.arcsin(clipped[wire]), wires=wire)
      qml.RZ(qml.math.arccos(clipped[wire]), wires=wire)

    for layer in range(q_layers):
      for wire in range(n_qubits):
        qml.RY(weights[layer, wire, 0], wires=wire)
        qml.RZ(weights[layer, wire, 1], wires=wire)
      for wire in range(n_qubits - 1):
        qml.CNOT(wires=[wire, wire + 1])
        qml.CZ(wires=[wire, wire + 1])

    return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

  q_weights = weights_dict["q_weights"]
  quantum_features = np.array([pure_numpy_qnn_circuit(sample, q_weights) for sample in quantum_inputs], dtype=np.float32)

  # 4. Process fusion array layers through the Sequential Head
  fused_features = np.concatenate([classical_features, quantum_features], axis=1)
  
  fc1_w = weights_dict["output_head.0.weight"]
  fc1_b = weights_dict["output_head.0.bias"]
  fc2_w = weights_dict["output_head.2.weight"]
  fc2_b = weights_dict["output_head.2.bias"]

  # Manual ReLU implementation matching training sequential layer
  layer1_out = np.maximum(0, np.dot(fused_features, fc1_w.T) + fc1_b)
  logits = np.dot(layer1_out, fc2_w.T) + fc2_b

  # CRITICAL FIX FOR BINARY CLASSIFIERS: Squashing logit vector via sigmoid
  probabilities = 1.0 / (1.0 + np.exp(-logits))

  prediction_results = []
  for p in probabilities.flatten():
    prob_val = float(p)
    prediction_results.append({
        "probability": prob_val,
        "prediction": 1 if prob_val >= 0.5 else 0
    })
      
  return prediction_results