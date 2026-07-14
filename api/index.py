import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import pennylane as qml

app = FastAPI(title="Q-Net Finance Selective Model Backend")

SEQUENCE_LENGTH = 5
WINDOW_DAYS = 10
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
  "Hybrid_QNN1_binary": load_model_weights("hybrid_qnn1_binary_weights.npy"),
  "Hybrid_QNN2_binary": load_model_weights("hybrid_qnn2_binary_weights.npy"),
}

def load_preprocessor(preprocessor_file_name: str):
  try:
    # 1. Load the joblib file
    JOBLIB_PATH = CURRENT_DIR / "preprocessors" / preprocessor_file_name
    preprocessor = joblib.load(JOBLIB_PATH)

    FEATURE_CANDIDATES = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsi_14",
        "macd",
        "macd_signal",
        "adx_14",
    )

    default_candidate_features = [
        feature for feature in FEATURE_CANDIDATES
    ]
    
    # 2. Extract components dynamically based on how your teammate structured it
    # Case A: If it's saved as a dictionary:
    if isinstance(preprocessor, dict):
      selector = preprocessor.get("selector")
      x_scaler = preprocessor.get("x_scaler")
      y_scaler = preprocessor.get("y_scaler")
      selected_features = preprocessor.get("selected_features")
      candidate_features = preprocessor.get("candidate_features", default_candidate_features)
      lookback = preprocessor.get("lookback")
      feature_range = preprocessor.get("feature_range")
      sequence_length = preprocessor.get("lookback", WINDOW_DAYS) 
        
    # Case B: If it's a custom wrapper class object (e.g., PreparedData or Preprocessor):
    else:
      selector = getattr(preprocessor, "selector", None)
      x_scaler = getattr(preprocessor, "x_scaler", None)
      y_scaler = getattr(preprocessor, "y_scaler", None)
      selected_features = getattr(preprocessor, "selected_features", None)
      candidate_features = getattr(preprocessor, "candidate_features", default_candidate_features)
      feature_range = getattr(preprocessor, "feature_range", None)
      lookback = getattr(preprocessor, "lookback", None)
      sequence_length = getattr(preprocessor, "lookback", WINDOW_DAYS)

    return selector, x_scaler, y_scaler, selected_features, candidate_features, lookback, feature_range, sequence_length
  except Exception as e:
    print(f"Error loading preprocessor.joblib: {e}")

PREPROCESSORS = {
  "Hybrid_QNN": load_preprocessor("hybrid_qnn.joblib"),
  "Hybrid_QNN_binary": load_preprocessor("hybrid_qnn_binary.joblib"),
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

def forward_numpy_lstm(X: np.ndarray, w: dict) -> np.ndarray:
  """
  A pure NumPy implementation of standard LSTM sequence processing to completely replace torch.
  Assumes standard single or multi-layer weights exported from a PyTorch LSTM state_dict.
  """
  batch_size, seq_len, _ = X.shape

  
  # Simple extraction of weight matrices (assumes a single-layer LSTM for readability)
  # If using multi-layer, loop across your weight keys accordingly
  w_ih = w["extractor.lstm.weight_ih_l0"] # [4*hidden_size, input_size]
  w_hh = w["extractor.lstm.weight_hh_l0"] # [4*hidden_size, hidden_size]
  b_ih = w["extractor.lstm.bias_ih_l0"]   # [4*hidden_size]
  b_hh = w["extractor.lstm.bias_hh_l0"]   # [4*hidden_size]
  
  hidden_size = w_hh.shape[1]
  
  # Extract linear projection head mapping hidden states down to qubit features
  w_proj = w["extractor.qubit_projection.weight"] # [n_qubits, hidden_size]
  b_proj = w["extractor.qubit_projection.bias"]   # [n_qubits]
  
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

def run_hybrid1_bulk_model_inference(
    weights_dict: dict,        # Pre-loaded dict containing numpy arrays from your model
    raw_ohlcv_list: list, 
    selector, 
    x_scaler, 
    y_scaler, 
    candidate_features: list,
    sequence_length: int,
    n_qubits: int,
    q_layers: int
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
  features_df = df[candidate_features] if candidate_features else df

  # 1. Apply rolling average
  for column in candidate_features:
    values = features_df[column].to_numpy(dtype=np.float64)
    features_df[column] = pd.Series(values).rolling(window=WINDOW_DAYS, min_periods=1).mean().to_numpy(dtype=np.float64)
    
  # 2. Apply the pre-fitted SelectKBest transformation and MinMaxScaler 
  selected_features_matrix = selector.transform(features_df)
  X_scaled = x_scaler.transform(selected_features_matrix)
  row_count = len(X_scaled)
 
  # 3. Construct chronological overlapping window sequences
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

  # 4. Extract weights from your pre-loaded parameters dict
  # Extract structural numpy values directly from your state dict keys
  q_weights = weights_dict["q_weights"]         # Shape: [q_layers, n_qubits, 2]
  w_head = weights_dict["output_head.weight"]   # Shape: [1, n_qubits]
  b_head = weights_dict["output_head.bias"]     # Shape: [1,]

  # 5. Define standard NumPy-native PennyLane Circuit Loop
  dev = qml.device("default.qubit", wires=n_qubits)

  @qml.qnode(dev, interface=None) # interface=None enforces native NumPy execution
  def pure_numpy_qnn_circuit(features, weights):
    clipped = np.clip(features, -0.999999, 0.999999)
    for wire in range(n_qubits):
      qml.RY(np.arcsin(clipped[wire]), wires=wire)
      qml.RZ(np.arccos(clipped[wire]), wires=wire)

    for layer in range(q_layers):
      for wire in range(n_qubits):
        qml.RY(weights[layer, wire, 0], wires=wire)
        qml.RZ(weights[layer, wire, 1], wires=wire)
      for wire in range(n_qubits - 1):
        qml.CNOT(wires=[wire, wire + 1])
        qml.CZ(wires=[wire, wire + 1])

    return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

  try:
    # 6. NumPy LSTM Forward Pass Integration
    # We process the arrays via standard manual recurrent loops
    qubit_features = forward_numpy_lstm(X_array, weights_dict)
  
    # 7. Evaluate states via PennyLane NumPy loop
    quantum_features = []
    for sample in qubit_features:
      q_res = pure_numpy_qnn_circuit(sample, q_weights)
      quantum_features.append(q_res)
        
    q_features = np.array(quantum_features, dtype=np.float32) # [Batch, n_qubits]
    
    # 8. Classical Output Head Evaluation via Matrix Vector Multiplication
    predictions_scaled = np.dot(q_features, w_head.T) + b_head
      
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error running inference loop: {str(e)}")
      
  # 9. Reverse scaled forecasts back to true market dollar values
  predictions_unscaled = y_scaler.inverse_transform(predictions_scaled).flatten().tolist()
  return [float(p) for p in predictions_unscaled]

def run_hybrid2_bulk_model_inference(
    weights_dict: dict,        # Pre-loaded dict containing numpy arrays from your model
    raw_ohlcv_list: list, 
    selector, 
    x_scaler, 
    y_scaler, 
    candidate_features: list,
    sequence_length: int,
    n_qubits: int,
    q_layers: int
) -> list:
    """
    Inference pipeline for HybridQNN2 executing entirely without PyTorch.
    Data scaling -> Sequential window formatting -> Dual-stream evaluation ->
    Feature fusion -> Multi-layer regression head -> Inverse Scaler
    """
    if not raw_ohlcv_list or len(raw_ohlcv_list) < sequence_length:
        return []
        
    # 1. Transform raw list into technical indicators 
    df = pd.DataFrame(raw_ohlcv_list)
    df = canonicalize_ohlcv_columns(df)  
    df = add_technical_indicators(df).dropna()
    
    if len(df) == 0:
        return []

    features_df = df[candidate_features]
    
    # 2. Select matching features and scale down to [-1.0, 1.0] bounds
    selected_features_matrix = selector.transform(features_df)
    X_scaled = x_scaler.transform(selected_features_matrix)
    row_count = len(X_scaled)

    # 3. Construct chronological overlapping window sequences [Batch, Sequence, Features]
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
    batch_size = X_array.shape[0]

    # 4. Extract standard parameters from saved weights dictionary
    q_weights = weights_dict["q_weights"]                                 # [q_layers, n_qubits, 2]
    q_proj_w = weights_dict["quantum_projection.weight"]                  # [n_qubits, lookback * n_features]
    q_proj_b = weights_dict["quantum_projection.bias"]                    # [n_qubits]
    
    # Output Head Parameters (Sequential Layer: Linear -> ReLU -> Linear)
    fc1_w = weights_dict["output_head.0.weight"]                         # [fusion_hidden, lstm_hidden + n_qubits]
    fc1_b = weights_dict["output_head.0.bias"]                           # [fusion_hidden]
    fc2_w = weights_dict["output_head.2.weight"]                         # [1, fusion_hidden]
    fc2_b = weights_dict["output_head.2.bias"]                           # [1]

    # 5. Define standard NumPy-native PennyLane Circuit Loop
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface=None) # Native NumPy execution without torch bindings
    def pure_numpy_qnn_circuit(features, weights):
        clipped = np.clip(features, -0.999999, 0.999999)
        for wire in range(n_qubits):
            qml.RY(np.arcsin(clipped[wire]), wires=wire)
            qml.RZ(np.arccos(clipped[wire]), wires=wire)

        for layer in range(q_layers):
            for wire in range(n_qubits):
                qml.RY(weights[layer, wire, 0], wires=wire)
                qml.RZ(weights[layer, wire, 1], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
                qml.CZ(wires=[wire, wire + 1])

        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

    try:
        # ------------------ CLASSICAL STREAM (LSTM) ------------------
        # Run standard numpy-based forward propagation through the LSTM module
        classical_features = forward_numpy_lstm_only(X_array, weights_dict) # [Batch, lstm_hidden_size]
        
        # ------------------ QUANTUM STREAM (CustomQNN) ----------------
        # Flatten input: [Batch, Sequence, Features] -> [Batch, Sequence * Features]
        flattened_inputs = X_array.reshape(batch_size, -1)
        
        # Project using linear mapping and scale via tanh activation
        quantum_inputs = np.tanh(np.dot(flattened_inputs, q_proj_w.T) + q_proj_b) # [Batch, n_qubits]
        
        # Execute individual quantum circuits
        quantum_features_list = []
        for sample in quantum_inputs:
            q_res = pure_numpy_qnn_circuit(sample, q_weights)
            quantum_features_list.append(q_res)
            
        quantum_features = np.array(quantum_features_list, dtype=np.float32) # [Batch, n_qubits]
        
        # ------------------ FEATURE FUSION & PREDICTION ----------------
        # Concatenate classical and quantum feature representations along dimension 1
        fused_features = np.concatenate([classical_features, quantum_features], axis=1) # [Batch, lstm_hidden + n_qubits]
        
        # First linear layer + ReLU activation function: max(0, xW_1^T + b_1)
        layer1_out = np.maximum(0, np.dot(fused_features, fc1_w.T) + fc1_b)
        
        # Second linear mapping to output pricing predictions
        predictions_scaled = np.dot(layer1_out, fc2_w.T) + fc2_b # [Batch, 1]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running HybridQNN2 inference: {str(e)}")
        
    # 6. Inverse scale predictions back to normal market prices
    predictions_unscaled = y_scaler.inverse_transform(predictions_scaled).flatten().tolist()
    return [float(p) for p in predictions_unscaled]


def forward_numpy_lstm_only(X: np.ndarray, w: dict) -> np.ndarray:
    """
    Pure NumPy implementation of PyTorch LSTM to extract the last step's hidden state.
    Supports single or multi-layer weights exported from a standard PyTorch state_dict.
    """
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
        #   "adjClose": float(adjClose)
        } for open, high, low, close, volume, adjClose in zip(cleaned_opens.split(","), cleaned_highs.split(","), cleaned_lows.split(","), cleaned_closes.split(","), cleaned_volumes.split(","), cleaned_adjCloses.split(","))
      ]
      selector, x_scaler, y_scaler, selected_features, candidate_features, lookback, feature_range, sequence_length = PREPROCESSORS.get('Hybrid_QNN')
      if model_name == "Hybrid_QNN1":
        predictions_list = run_hybrid1_bulk_model_inference(target_weights, ohlcv_list, selector, x_scaler, y_scaler, candidate_features, sequence_length, 3, 1) 
        results[model_name] = predictions_list
      else:
        predictions_list = run_hybrid2_bulk_model_inference(target_weights, ohlcv_list, selector, x_scaler, y_scaler, candidate_features, sequence_length, 3, 1) 
        results[model_name] = predictions_list

  return {
    "status": "success",
    "predictions": results
  }