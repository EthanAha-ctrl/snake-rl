import cv2
import lmdb
import pickle
import numpy as np
import os
import random

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
LMDB_PATH = os.path.join(DATA_ROOT, "coc_train.lmdb")
META_PATH = os.path.join(DATA_ROOT, "coc_meta.pkl")

def main():
    if not os.path.exists(LMDB_PATH) or not os.path.exists(META_PATH):
        print(f"Error: Dataset files not found at {DATA_ROOT}")
        print("Please run preprocessing.py first.")
        return

    print("Loading metadata...")
    with open(META_PATH, 'rb') as f:
        meta_info = pickle.load(f)
    print(f"Loaded {len(meta_info)} samples.")

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

                # Display
                title = f"R: {label} | Idx: {current_idx}/{total_samples}"
                cv2.imshow("CoC Dataset visualizer", img)
                cv2.setWindowTitle("CoC Dataset visualizer", title)
                
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
