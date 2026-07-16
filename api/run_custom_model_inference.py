import numpy as np

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import PauliProductRotationGate
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN

from typing import Literal, Sequence, Dict, Any

DEFAULT_NUM_QUBITS = 5
DEFAULT_NUM_LAYERS = 1

PROJECT_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "SMA5",
    "ADX",
    "Return_1",
]

def canonicalize_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize case variants such as `Close` and `Volume` to lower case."""
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [str(column[0]) for column in frame.columns]

    renamed: Dict[Any, str] = {}
    for column in frame.columns:
        normalized = str(column).strip()[0].upper() + str(column).strip()[1:].lower()
        if normalized in {"Open", "High", "Low", "Close", "Volume"}:
            renamed[column] = normalized
    output = frame.rename(columns=renamed).copy()

    required = {"Open", "High", "Low", "Close"}
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
    """Add RSI, MACD, SMA5, ADX, and lagged returns without extra TA packages."""

    data = frame.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    data["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["SMA5"] = close.rolling(window=5, min_periods=5).mean()
    data["Return_1"] = close.pct_change()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=14, min_periods=14).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=data.index).rolling(14).sum() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=data.index).rolling(14).sum() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    data["ADX"] = dx.rolling(window=14, min_periods=14).mean()
    return data

def ppr_gate(width: int, angle: float | Parameter = 0) -> PauliProductRotationGate:
    return PauliProductRotationGate(Pauli("Z" * width), angle=angle, label="Ppr")

def _append_rotation(
    circuit: QuantumCircuit,
    axis: Literal["rx", "ry", "rz"],
    qubit: int,
    weight_params: list[Parameter] | None,
) -> None:
    """Append a zero or trainable rotation without changing operation order."""

    angle: float | Parameter
    if weight_params is None:
        angle = 0
    else:
        angle = Parameter(f"theta_{len(weight_params):03d}_{axis}_q{qubit}")
        weight_params.append(angle)
    getattr(circuit, axis)(angle, qubit)

def _append_ppr(
    circuit: QuantumCircuit,
    qubits: Sequence[int],
    weight_params: list[Parameter] | None,
) -> None:
    """Append a fixed or trainable Pauli Product Rotation block."""

    if weight_params is None:
        angle: float | Parameter = 0
    else:
        angle = Parameter(f"theta_{len(weight_params):03d}_ppr_w{len(qubits)}")
        weight_params.append(angle)
    circuit.append(ppr_gate(len(qubits), angle), list(qubits))

def build_parameterized_qnn_regressor_circuit(
    num_qubits: int = DEFAULT_NUM_QUBITS,
) -> tuple[QuantumCircuit, list[Parameter]]:
    """Transcribe Fig. 6 with trainable parameters at existing rotation gates."""

    circuit, weight_params = _build_qnn_regressor_circuit(num_qubits, parameterized=True)
    circuit.name = "parameterized_qnn_regressor"
    return circuit, weight_params

def _build_qnn_regressor_circuit(
    num_qubits: int,
    parameterized: bool,
) -> tuple[QuantumCircuit, list[Parameter]]:
    """Shared Fig. 6 builder used for original and trainable circuits."""

    if num_qubits != DEFAULT_NUM_QUBITS:
        raise ValueError("The manually reproduced Fig. 6 circuit is a five-qubit circuit.")

    weights: list[Parameter] | None = [] if parameterized else None
    circuit = QuantumCircuit(num_qubits)
    rot = lambda axis, qubit: _append_rotation(circuit, axis, qubit, weights)

    # Gate order copied from reproduce_quantum_circuit.py::build_qnn_regressor_circuit.
    circuit.h([0, 1, 2, 3, 4])
    circuit.cx(1, 2)
    circuit.cx(0, 4)
    rot("rz", 2)
    rot("rz", 4)
    circuit.cx(1, 2)
    circuit.cx(0, 4)
    rot("ry", 0)
    rot("ry", 4)
    circuit.cx(1, 3)
    rot("rz", 0)
    rot("rz", 3)
    rot("rz", 4)
    rot("rx", 0)
    circuit.cx(1, 3)
    rot("rz", 0)
    rot("ry", 1)
    circuit.cx(2, 3)
    rot("rz", 1)
    rot("rz", 3)
    _append_ppr(circuit, [0, 1], weights)
    circuit.cx(2, 3)
    rot("rx", 1)
    rot("ry", 2)
    rot("ry", 3)
    rot("rz", 1)
    rot("rz", 2)
    rot("rz", 3)
    _append_ppr(circuit, [1, 2], weights)
    rot("rx", 2)
    rot("rz", 2)
    _append_ppr(circuit, [2, 3], weights)
    rot("ry", 2)
    rot("ry", 2)
    rot("rx", 3)
    rot("rz", 3)
    _append_ppr(circuit, [3, 4], weights)
    rot("rz", 2)
    rot("rz", 2)
    rot("rx", 4)
    rot("rz", 4)
    circuit.cx(2, 3)
    _append_ppr(circuit, [0, 1, 2, 3, 4], weights)
    rot("ry", 0)
    rot("ry", 3)
    rot("ry", 4)

    # Continuation line of the folded drawing.
    rot("ry", 0)
    rot("rz", 0)
    rot("rz", 0)
    circuit.cx(0, 1)
    rot("ry", 1)
    rot("ry", 1)
    rot("rz", 1)
    rot("rz", 1)
    rot("ry", 3)
    rot("rz", 3)
    rot("rz", 3)
    rot("ry", 4)
    rot("rz", 4)
    rot("rz", 4)

    return circuit, ([] if weights is None else weights)

def build_parameterized_angle_encoding_circuit(
    num_qubits: int = DEFAULT_NUM_QUBITS,
    input_mode: Literal["single", "dual"] = "single",
) -> tuple[QuantumCircuit, list[Parameter]]:
    """Build the parameterized Fig. 4 encoding circuit.

    ``single`` mode exposes one input angle per qubit and uses it in both the
    Ry/Rz encoding locations. This keeps the QNN input dimension equal to the
    selected financial feature dimension.

    ``dual`` mode exposes separate theta and phi parameters for Ry/Rz, matching
    the two-angle Fig. 4 formula directly. Use this if preprocessing expands
    each feature into ``arcsin(x)`` and ``arccos(x)`` angles.
    """

    if input_mode not in {"single", "dual"}:
        raise ValueError("input_mode must be either 'single' or 'dual'.")

    size = num_qubits if input_mode == "single" else 2 * num_qubits
    input_vector = ParameterVector("x", size)
    input_params = list(input_vector)

    circuit = QuantumCircuit(num_qubits, name="parameterized_angle_encoding")
    for qubit in range(num_qubits):
        circuit.ry(input_vector[qubit], qubit)
        rz_param = input_vector[qubit] if input_mode == "single" else input_vector[num_qubits + qubit]
        circuit.rz(rz_param, qubit)
    return circuit, input_params

def build_custom_qnn_circuit(
    num_qubits: int = DEFAULT_NUM_QUBITS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    input_mode: Literal["single", "dual"] = "single",
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Return a trainable QNN circuit plus input and weight parameters.

    The function preserves the existing project circuit. ``num_layers`` is kept
    in the signature for project compatibility, but only ``1`` is accepted until
    a deliberate new repeated-layer architecture is approved.
    """

    if num_layers != 1:
        raise ValueError(
            "num_layers must be 1 to preserve the current manually rebuilt circuit. "
            "Using more layers would repeat gates and change the architecture."
        )

    encoding, input_params = build_parameterized_angle_encoding_circuit(
        num_qubits=num_qubits,
        input_mode=input_mode,
    )
    regressor, weight_params = build_parameterized_qnn_regressor_circuit(num_qubits)
    circuit = encoding.compose(regressor)
    circuit.name = "custom_trainable_qnn"
    return circuit, input_params, weight_params

def z_observable(num_qubits: int, output_qubit: int = 0) -> SparsePauliOp:
    """Create a one-output Pauli-Z observable for regression."""

    if output_qubit < 0 or output_qubit >= num_qubits:
        raise ValueError("output_qubit is out of range.")
    label = ["I"] * num_qubits
    label[num_qubits - 1 - output_qubit] = "Z"
    return SparsePauliOp.from_list([("".join(label), 1.0)])

def create_estimator_qnn(
    circuit: QuantumCircuit,
    input_params: Sequence[Parameter],
    weight_params: Sequence[Parameter],
    output_qubit: int = 0,
    input_gradients: bool = True,
) -> EstimatorQNN:
    observable = z_observable(circuit.num_qubits, output_qubit)
    return EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=list(input_params),
        weight_params=list(weight_params),
        input_gradients=input_gradients,
    )

def expectation_to_scaled_target(y_expectation: np.ndarray) -> np.ndarray:
    """Map QNN expectation values from [-1, 1] back to [0, 1]."""

    return (y_expectation + 1.0) / 2.0

def run_custom_qnn_inference(
    weights: np.ndarray, 
    features_batch: np.ndarray, 
    num_qubits: int = DEFAULT_NUM_QUBITS
) -> list[float]:
    # 1. Reconstruct the exact preserved circuit architecture
    circuit, input_params, weight_params = build_custom_qnn_circuit(num_qubits=num_qubits)
    
    # 2. Re-create the EstimatorQNN wrapper without PyTorch
    # We set input_gradients=False since we only need the forward pass for inference
    qnn = create_estimator_qnn(
        circuit=circuit,
        input_params=input_params,
        weight_params=weight_params,
        input_gradients=False
    )
    
    # 3. Execute the forward pass
    # EstimatorQNN evaluates the Pauli-Z observable and natively returns 
    # expectation values in the range [-1, 1].
    expectations = qnn.forward(input_data=features_batch, weights=weights)
    
    # 4. Map expectations [-1, 1] back to the MinMax scaled range [0, 1][cite: 5]
    predictions_scaled = expectation_to_scaled_target(expectations)
    
    # 5. Format results into a 2D list of shape (B, 1) for your backend router
    return predictions_scaled.reshape(-1, 1).tolist()

def run(
    weights_dict: dict,
    ohlcv_list: list, 
    selector, 
    x_scaler, 
    y_scaler,
    candidate_features: list,
    sequence_length: int,
  ) -> list[float]:
    ohlcv_df = pd.DataFrame(ohlcv_list)
    df_with_indicators = add_technical_indicators(canonicalize_ohlcv_columns(ohlcv_df))
    
    # df_valid = df_with_indicators.dropna(subset=PROJECT_FEATURES)
    df_valid = df_with_indicators.fillna(0)

    batch_rows = df_valid[PROJECT_FEATURES]
    
    scaled_features = x_scaler.transform(batch_rows)
    
    selected_features = selector.transform(scaled_features)
    
    # 5. Run inference on the entire batch at once
    predictions_scaled = run_custom_qnn_inference(
        weights=weights_dict['weight'], 
        features_batch=selected_features
    )
    
    predicted_prices = y_scaler.inverse_transform(predictions_scaled)
    
    return predicted_prices.flatten().tolist()