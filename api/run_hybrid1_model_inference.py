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

  q_weights = weights_dict["q_weights"]
  w_head = weights_dict["output_head.weight"]
  b_head = weights_dict["output_head.bias"]

  dev = qml.device("default.qubit", wires=n_qubits)

  @qml.qnode(dev, interface=None)
  def qnn_circuit(features, weights):
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

  try:
    qubit_features = forward_numpy_lstm(X_array, weights_dict)
    quantum_features = []
    for sample in qubit_features:
      q_res = qnn_circuit(sample, q_weights)
      quantum_features.append(q_res)
        
    q_features = np.array(quantum_features, dtype=np.float32) # [Batch, n_qubits]
    
    # 8. Classical Output Head Evaluation via Matrix Vector Multiplication
    predictions_scaled = np.dot(q_features, w_head.T) + b_head
      
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error running inference loop: {str(e)}")
      
  # 9. Reverse scaled forecasts back to true market dollar values
  predictions_unscaled = y_scaler.inverse_transform(predictions_scaled).flatten().tolist()
  return [float(p) for p in predictions_unscaled]
