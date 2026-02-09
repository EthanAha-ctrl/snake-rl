import cv2
import lmdb
import pickle
import numpy as np
import os
import torch
import torch.nn.functional as F
from model import get_hrnet_w18

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
LMDB_PATH = os.path.join(DATA_ROOT, "coc_train.lmdb")
META_PATH = os.path.join(DATA_ROOT, "coc_meta.pkl")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    if not os.path.exists(LMDB_PATH) or not os.path.exists(META_PATH):
        print(f"Error: Dataset files not found at {DATA_ROOT}")
        print("Please run preprocessing.py first.")
        return

    print("Loading metadata...")
    with open(META_PATH, 'rb') as f:
        meta_info = pickle.load(f)
    print(f"Loaded {len(meta_info)} samples.")

    # --- Load Model ---
    print(f"Loading model from {CHECKPOINT_PATH}...")
    # Initialize model structure (Must match train.py)
    model = get_hrnet_w18(num_classes=10, in_channels=1)
    
    # Load weights
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    else:
        print("Error: Checkpoint not found! Running with random weights (for testing logic only).")
    
    model.to(DEVICE)
    model.eval()

    print("Opening LMDB environment...")
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)

    print("Starting visualization loop. Press 'q' to quit, 'b' for previous, any other key for next.")
    
    current_idx = 0
    # The dataset was generated as: Source Image -> 10 Radius variants
    # Stored sequentially in meta_info.
    # We simply iterate through them.
    
    total_samples = len(meta_info)

    try:
        with env.begin() as txn:
            while True:
                if current_idx < 0: current_idx = 0
                if current_idx >= total_samples: current_idx = 0
                
                # Direct mapping since dataset is sequential (Image -> Radii)
                physical_idx = current_idx
                
                if physical_idx >= len(meta_info):
                    physical_idx = len(meta_info) - 1
                
                # Get sequential sample
                key_str, label = meta_info[physical_idx]
                
                # Retrieve image from LMDB
                img_bytes = txn.get(key_str.encode('ascii'))
                
                if img_bytes is None:
                    print(f"Warning: Key {key_str} missing.")
                    current_idx += 1
                    continue
                
                # Decode image
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"Warning: Failed to decode key {key_str}")
                    current_idx += 1
                    continue

                # --- Inference Logic ---
                # Preprocess: [H, W] -> float32 0-1 -> [1, 1, H, W]
                input_tensor = img.astype(np.float32) / 255.0
                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(input_tensor)
                    # Output shape: [1, 10, H, W] (Dense prediction)
                    # Ensure we handle spatial dimensions correctly.
                    # We average the logits over the spatial dimensions to get a global prediction vector [1, 10]
                    logits_avg = output.mean(dim=(2, 3)) 
                    pred_idx = logits_avg.argmax(1).item() # 0-9
                
                pred_radius = pred_idx
                
                # Display
                # Calculate error for display coloring (optional, but nice)
                diff = abs(pred_radius - label)
                status = "OK" if diff == 0 else f"Err: {diff}"
                
                title = f"GT: {label} | Pred: {pred_radius} ({status}) | Idx: {current_idx}/{total_samples}"
                
                cv2.imshow("CoC Inference Visualizer", img)
                cv2.setWindowTitle("CoC Inference Visualizer", title)
                
                # Wait for key
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('b'): # Back
                    current_idx -= 1
                else: # Next
                    current_idx += 1
    finally:
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
