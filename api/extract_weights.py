# Do this locally with torch. Torch will not be installed on the server to reduce size
import torch
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

for model in ["qlstm", "custom_qnn", "hybrid_qnn1"]:
    artifact = torch.load(CURRENT_DIR / "artifacts" / f"{model}.pt", map_location="cpu")
    state_dict = artifact["model_state_dict"] if "model_state_dict" in artifact else artifact['state_dict']
    np.save(CURRENT_DIR / "weights" / f"{model}_weights.npy", state_dict)
