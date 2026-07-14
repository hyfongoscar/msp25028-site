import numpy as np
from fastapi import HTTPException

def run(state_dict: dict, raw_prices: list, sequence_length: int) -> list:
  if not raw_prices or len(raw_prices) < sequence_length:
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
    if i >= sequence_length - 1:
      window = scaled_prices[i - sequence_length + 1 : i + 1]
    else:
      # Padding replication using the first element if the array is short at index 0
      padding = np.repeat(scaled_prices[0], sequence_length - i - 1)
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
