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

for model in ["qlstm", "custom_qnn", "hybrid_qnn1"]:
  artifact = torch.load(CURRENT_DIR / "artifacts" / f"{model}.pt", map_location="cpu")
  state_dict = artifact["model_state_dict"] if "model_state_dict" in artifact else artifact['state_dict']

  cleaned_weights = {}

  for key, value in state_dict.items():
      cleaned_weights[key] = clean_item(value)

  # 3. Save it back out over the old file
  np.save(CURRENT_DIR / "weights" / f"{model}_weights.npy", cleaned_weights, allow_pickle=True)
  print(f"Cleaned weights file successfully generated for {model} model!")