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
    model = get_hrnet_w18(num_classes=50, in_channels=1)
    
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
    # The dataset was generated as: For each Radius 1..50: For each Crop 1..5: Write
    # We want to visualize: For each Crop 1..5: For each Radius 1..50
    # Total variants per image = 50 * 5 = 250
    # Let's map "Logical Index" -> "Physical Index"
    
    def get_physical_index(logical_idx):
        # 1. Which Source Image? (Every 250 images is a new source image)
        img_block = logical_idx // 250
        within_block = logical_idx % 250
        
        # 2. Within Block:
        # Logical: Crop (0-4) varies slowly, Radius (0-49) varies fast
        logical_crop = within_block // 50
        logical_radius_idx = within_block % 50 
        
        # Physical: Radius (0-49) varies slowly, Crop (0-4) varies fast
        # physical_idx = (RadiusIdx * 5) + CropIdx
        physical_offset = (logical_radius_idx * 5) + logical_crop
        
        return (img_block * 250) + physical_offset

    total_samples = len(meta_info)

    try:
        with env.begin() as txn:
            while True:
                if current_idx < 0: current_idx = 0
                if current_idx >= total_samples: current_idx = 0
                
                # Use mapping
                physical_idx = get_physical_index(current_idx)
                
                # Careful about boundary (if total samples isn't multiple of 250, simple logic might break at end, but fine for now)
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
                    # Output shape: [1, 50, H, W] (Dense prediction)
                    # Ensure we handle spatial dimensions correctly.
                    # We average the logits over the spatial dimensions to get a global prediction vector [1, 50]
                    logits_avg = output.mean(dim=(2, 3)) 
                    pred_idx = logits_avg.argmax(1).item() # 0-49
                
                pred_radius = pred_idx + 1 # Convert back to 1-50
                
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
