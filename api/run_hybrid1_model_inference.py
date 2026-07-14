import numpy as np
import pandas as pd
import pennylane as qml
from typing import Dict, Any
from fastapi import HTTPException

WINDOW_DAYS = 10

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

def forward_numpy_lstm(X: np.ndarray, w: dict) -> np.ndarray:
  batch_size, seq_len, _ = X.shape

  w_ih = w["extractor.lstm.weight_ih_l0"]
  w_hh = w["extractor.lstm.weight_hh_l0"]
  b_ih = w["extractor.lstm.bias_ih_l0"]
  b_hh = w["extractor.lstm.bias_hh_l0"]
  
  hidden_size = w_hh.shape[1]
  
  w_proj = w["extractor.qubit_projection.weight"]
  b_proj = w["extractor.qubit_projection.bias"]
  
  final_qubit_features = []
    
  for b in range(batch_size):
    h = np.zeros(hidden_size)
    c = np.zeros(hidden_size)
    
    # Chronological recurrence tracking
    for t in range(seq_len):
      x_t = X[b, t, :]
      
      # Traditional gate calculations (Input, Forget, Cell, Output)
      gates = np.dot(w_ih, x_t) + b_ih + np.dot(w_hh, h) + b_hh
      i, f, g, o = np.split(gates, 4)
      
      # Activation applications
      i = 1.0 / (1.0 + np.exp(-i))
      f = 1.0 / (1.0 + np.exp(-f))
      g = np.tanh(g)
      o = 1.0 / (1.0 + np.exp(-o))
      
      c = f * c + i * g
      h = o * np.tanh(c)
        
    # Map last hidden state vector to feature selected dimensions
    qubit_out = np.dot(w_proj, h) + b_proj
    final_qubit_features.append(qubit_out)

  return np.array(final_qubit_features)

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
  df = add_technical_indicators(df)
  
  features_df = df[candidate_features].copy()
  for column in features_df.columns:
    values = features_df[column].to_numpy(dtype=np.float64)
    features_df[column] = pd.Series(values).rolling(window=WINDOW_DAYS, min_periods=1).mean().to_numpy(dtype=np.float64)
      
  X_scaled = x_scaler.transform(selector.transform(features_df))
  
  sequences = []
  for i in range(len(X_scaled)):
    if i >= sequence_length - 1:
      window = X_scaled[i - sequence_length + 1 : i + 1]
    else:
      window = np.concatenate([np.repeat(X_scaled[0:1], sequence_length - i - 1, axis=0), X_scaled[0 : i + 1]], axis=0)
    sequences.append(window)
      
  X_array = np.array(sequences, dtype=np.float32)
  
  batch_size, seq_len, n_feats = X_array.shape
  hidden_size = 16

  w_ih = weights_dict["extractor.lstm.weight_ih_l0"]
  w_hh = weights_dict["extractor.lstm.weight_hh_l0"]
  b_ih = weights_dict["extractor.lstm.bias_ih_l0"]
  b_hh = weights_dict["extractor.lstm.bias_hh_l0"]
  
  w_proj = weights_dict["extractor.qubit_projection.weight"]
  b_proj = weights_dict["extractor.qubit_projection.bias"]

  qubit_features = []
  
  # Process sequence evaluation manually per sample batch element
  for b in range(batch_size):
    h = np.zeros(hidden_size, dtype=np.float32)
    c = np.zeros(hidden_size, dtype=np.float32)
    
    for t in range(seq_len):
      x_t = X_array[b, t, :] # [n_feats]
      
      # PyTorch LSTM Gate Math: i, f, g, o
      gates = np.dot(w_ih, x_t) + b_ih + np.dot(w_hh, h) + b_hh
      i = 1.0 / (1.0 + np.exp(-gates[0:hidden_size]))
      f = 1.0 / (1.0 + np.exp(-gates[hidden_size:2*hidden_size]))
      g = np.tanh(gates[2*hidden_size:3*hidden_size])
      o = 1.0 / (1.0 + np.exp(-gates[3*hidden_size:4*hidden_size]))
      
      c = f * c + i * g
      h = o * np.tanh(c)
        
    # Project the final hidden state layer
    q_feat = np.dot(w_proj, h) + b_proj
    qubit_features.append(q_feat)

  qubit_features = np.array(qubit_features, dtype=np.float32)

  dev = qml.device("default.qubit", wires=n_qubits)
  @qml.qnode(dev, interface=None)
  def pure_numpy_qnn_circuit(features, weights):
    safe_features = np.tanh(features)
    clipped = qml.math.clip(safe_features, -0.999999, 0.999999)
    
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
  quantum_features = np.array([pure_numpy_qnn_circuit(sample, q_weights) for sample in qubit_features], dtype=np.float32)
  
  # Linear execution head mapping
  w_head = weights_dict["output_head.weight"]
  b_head = weights_dict["output_head.bias"]
  predictions_scaled = np.dot(quantum_features, w_head.T) + b_head
  
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
  df = pd.DataFrame(ohlcv_list)
  df = canonicalize_ohlcv_columns(df)
  df = add_technical_indicators(df)
  
  if len(df) == 0:
      return []

  # Clean data slice to prevent SettingWithCopy errors
  features_df = df[candidate_features].copy() if candidate_features else df.copy()

  # Apply rolling filter matching training pipeline specs
  for column in features_df.columns:
      values = features_df[column].to_numpy(dtype=np.float64)
      features_df[column] = pd.Series(values).rolling(window=WINDOW_DAYS, min_periods=1).mean().to_numpy(dtype=np.float64)
      
  # Scale inputs sequentially
  selected_features_matrix = selector.transform(features_df)
  X_scaled = x_scaler.transform(selected_features_matrix)
  row_count = len(X_scaled)
  
  # Reconstruct chronological temporal arrays
  sequences = []
  for i in range(row_count):
    if i >= sequence_length - 1:
      window = X_scaled[i - sequence_length + 1 : i + 1]
    else:
      padding_count = sequence_length - i - 1
      padding = np.repeat(X_scaled[0:1], padding_count, axis=0)
      window = np.concatenate([padding, X_scaled[0 : i + 1]], axis=0)
    sequences.append(window)
      
  X_array = np.array(sequences, dtype=np.float32) # [Batch, Sequence, Features]
  batch_size, seq_len, n_feats = X_array.shape
  hidden_size = 16 
  
  # Extract structural LSTM parameters matching PyTorch gate layout order: i, f, g, o
  w_ih = weights_dict["extractor.lstm.weight_ih_l0"] 
  w_hh = weights_dict["extractor.lstm.weight_hh_l0"] 
  b_ih = weights_dict["extractor.lstm.bias_ih_l0"]
  b_hh = weights_dict["extractor.lstm.bias_hh_l0"]
  
  w_proj = weights_dict["extractor.qubit_projection.weight"] 
  b_proj = weights_dict["extractor.qubit_projection.bias"]   

  qubit_features = []
  
  # Propagate through sequential sequences element-by-element
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
        
    q_feat = np.dot(w_proj, h) + b_proj
    qubit_features.append(q_feat)

  qubit_features = np.array(qubit_features, dtype=np.float32)

  # Initialize Backend PennyLane Device context
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
  quantum_features = np.array([pure_numpy_qnn_circuit(sample, q_weights) for sample in qubit_features], dtype=np.float32)
  
  # Apply raw linear prediction head mappings to yield Logits
  w_head = weights_dict["output_head.weight"]
  b_head = weights_dict["output_head.bias"]
  logits = np.dot(quantum_features, w_head.T) + b_head
  
  # CRITICAL FIX FOR BINARY CLASSIFIERS: Run outputs through standard Sigmoid function
  probabilities = 1.0 / (1.0 + np.exp(-logits))
  
  prediction_results = []
  for p in probabilities.flatten():
    prob_val = float(p)
    prediction_results.append({
        "probability": prob_val,
        "prediction": 1 if prob_val >= 0.5 else 0
    })
      
  return prediction_results