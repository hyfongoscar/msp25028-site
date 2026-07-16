# Do this locally with torch. Torch will not be installed on the server to reduce size
import torch
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

def clean_item(item):
  # Convert PyTorch tensors to pure NumPy arrays
  if isinstance(item, torch.Tensor):
    return item.detach().cpu().numpy()
  elif isinstance(item, dict):
      return {k: clean_item(v) for k, v in item.items()}
  elif isinstance(item, list):
    return [clean_item(v) for v in item]
  return item

# @title
import torch
from torch import nn
import pennylane as qml

class QLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=4,
        n_qlayers=1,
        n_vrotations=3,
        batch_first=True,
        return_sequences=False,
        return_state=False,
        backend="default.qubit",
    ):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_vrotations = n_vrotations
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_q_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        # self.dev_forget = qml.device(self.backend, wires=self.n_qubits)
        # self.dev_input = qml.device(self.backend, wires=self.n_qubits)
        # self.dev_update = qml.device(self.backend, wires=self.n_qubits)
        # self.dev_output = qml.device(self.backend, wires=self.n_qubits)

        def ansatz(params, wires_type):
            # Entangling layer.
            for i in range(1, 3):
                for j in range(self.n_qubits):
                    if j + i < self.n_qubits:
                        qml.CNOT(wires=[wires_type[j], wires_type[j + i]])
                    else:
                        qml.CNOT(
                            wires=[wires_type[j], wires_type[j + i - self.n_qubits]]
                        )

            # Variational layer.
            for i in range(self.n_qubits):
                qml.RX(params[0][i], wires=wires_type[i])
                qml.RY(params[1][i], wires=wires_type[i])
                qml.RZ(params[2][i], wires=wires_type[i])

        def VQC(features, weights, wires_type):
            # Preproccess input data to encode the initial state.
            qml.templates.AngleEmbedding(features, wires=wires_type)
            ry_params = [torch.arctan(feature) for feature in features][0]
            rz_params = [torch.arctan(feature**2) for feature in features][0]
            for i in range(self.n_qubits):
                qml.Hadamard(wires=wires_type[i])
                qml.RY(ry_params[i], wires=wires_type[i])
                qml.RZ(rz_params[i], wires=wires_type[i])

            # Variational block.
            qml.layer(ansatz, self.n_qlayers, weights, wires_type=wires_type)

        def _circuit_forget(inputs, weights):
            VQC(inputs, weights, self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_forget]

        self.qlayer_forget = qml.QNode(
            _circuit_forget, self.dev_forget, interface="torch"
        )

        def _circuit_input(inputs, weights):
            VQC(inputs, weights, self.wires_input)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_input]

        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")

        def _circuit_q_update(inputs, weights):
            VQC(inputs, weights, self.wires_update)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_update]

        self.qlayer_q_update = qml.QNode(
            _circuit_q_update, self.dev_q_update, interface="torch"
        )

        def _circuit_output(inputs, weights):
            VQC(inputs, weights, self.wires_output)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_output]

        self.qlayer_output = qml.QNode(
            _circuit_output, self.dev_output, interface="torch"
        )

        weight_shapes = {"weights": (self.n_qlayers, self.n_vrotations, self.n_qubits)}
        print(
            f"weight_shapes = (n_qlayers, n_vrotations, n_qubits) = ({self.n_qlayers}, {self.n_vrotations}, {self.n_qubits})"
        )

        self.clayer_in = torch.nn.Linear(self.concat_size, self.n_qubits)
        self.VQC = nn.ModuleDict({
            "forget": qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            "input": qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            "q_update": qml.qnn.TorchLayer(self.qlayer_q_update, weight_shapes),
            "output": qml.qnn.TorchLayer(self.qlayer_output, weight_shapes),
        })
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        # self.clayer_out = [torch.nn.Linear(n_qubits, self.hidden_size) for _ in range(4)]

    def forward(self, x, init_states=None):
        """
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        """
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]

            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(
                self.clayer_out(self.VQC["forget"](y_t))
            )  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC["input"](y_t)))  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC["q_update"](y_t)))  # update block
            o_t = torch.sigmoid(
                self.clayer_out(self.VQC["output"](y_t))
            )  # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class QShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, n_qubits=0, n_qlayers=1):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = QLSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(
            hn
        ).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out

for model in ["qlstm"]:
  checkpoint = torch.load(CURRENT_DIR / "artifacts" / f"{model}.pt", map_location="cpu")

  model = QShallowRegressionLSTM(
      num_sensors=1,
      hidden_units=16,
      n_qubits=4,
      n_qlayers=1
  )

  # Handle both common ways PyTorch checkpoints are saved:
  # (Sometimes it's just the state_dict, sometimes it's a dict containing 'model_state_dict', epoch, loss, etc.)
  if 'model_state_dict' in checkpoint:
      model.load_state_dict(checkpoint['model_state_dict'])
  else:
      model.load_state_dict(checkpoint)

  # 3. Extract the weights and convert them to pure NumPy arrays
  weights_dict_np = {}
  for name, param in model.state_dict().items():
      # .detach() removes the tensor from the computational graph
      # .cpu() moves it to CPU memory
      # .numpy() converts it to a standard NumPy array
      weights_dict_np[name] = param.detach().cpu().numpy()

  # 4. Verify the final linear layer is included
  print("Extracted layers:")
  print(weights_dict_np.keys())

  # 5. Save the dictionary as a single .npy file
  save_path = 'qlstm_weights_epoch_50.npy'
  np.save(CURRENT_DIR / "weights" / f"{model}_weights.npy", weights_dict_np, allow_pickle=True)
  print(f"Weights successfully saved to {save_path}")


for model in []:
  artifact = torch.load(CURRENT_DIR / "artifacts" / f"{model}.pt", map_location="cpu")
  state_dict = artifact["model_state_dict"] if "model_state_dict" in artifact else artifact['state_dict']

  cleaned_weights = {}

  for key, value in state_dict.items():
      cleaned_weights[key] = clean_item(value)

  # 3. Save it back out over the old file
  np.save(CURRENT_DIR / "weights" / f"{model}_weights.npy", cleaned_weights, allow_pickle=True)
  print(f"Cleaned weights file successfully generated for {model} model!")