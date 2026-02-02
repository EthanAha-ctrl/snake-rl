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
    total_samples = len(meta_info)

    try:
        with env.begin() as txn:
            while True:
                if current_idx < 0: current_idx = 0
                if current_idx >= total_samples: current_idx = 0
                
                # Get sequential sample
                item = meta_info[current_idx]
                sharpness_grid = None
                
                if len(item) == 3:
                    key_str, label, sharpness_grid = item
                else:
                    key_str, label = item
                
                # Retrieve image from LMDB
                img_bytes = txn.get(key_str.encode('ascii'))
                
                if img_bytes is None:
                    print(f"Warning: Key {key_str} missing.")
                    current_idx += 1
                    continue
                
                # Decode image
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img_gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                
                if img_gray is None:
                    print(f"Warning: Failed to decode key {key_str}")
                    current_idx += 1
                    continue
                
                # Prepare display image (BGR)
                display_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                
                # Visualize Sharpness if available
                if sharpness_grid is not None:
                    # Find the base sample (sharpest) for the current sequence to fix the scale
                    # Assuming 50 variants per image as generated in preprocessing.py
                    VARIANTS_PER_IMAGE = 50
                    base_idx = (current_idx // VARIANTS_PER_IMAGE) * VARIANTS_PER_IMAGE
                    
                    # Retrieve base sharpness max value
                    try:
                        base_item = meta_info[base_idx]
                        if len(base_item) == 3:
                            _, _, base_sharpness = base_item
                            if base_sharpness is not None:
                                global_max = base_sharpness.max()
                            else:
                                global_max = sharpness_grid.max()
                        else:
                            global_max = sharpness_grid.max()
                    except IndexError: # Should not happen unless indices are messed up
                         global_max = sharpness_grid.max()
                    
                    if global_max < 1e-6: global_max = 1.0

                    # Normalize relative to the sharpest version
                    s_norm = sharpness_grid / global_max
                    s_norm = np.clip(s_norm, 0, 1)
                        
                    s_uint8 = (s_norm * 255).astype(np.uint8)
                    
                    # Resize to match image height (480)
                    h, w = img_gray.shape[:2]
                    # Target width for sharpness map to maintain aspect roughly or square pixels
                    # Grid is 20x15. Image is 640x480. 
                    # 640/20 = 32, 480/15 = 32. Perfect square blocks.
                    s_resized = cv2.resize(s_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Apply colormap
                    s_color = cv2.applyColorMap(s_resized, cv2.COLORMAP_JET)
                    
                    # Concatenate
                    display_img = np.hstack([display_img, s_color])

                # Display
                title = f"R: {label} | Idx: {current_idx}/{total_samples}"
                cv2.imshow("CoC Dataset visualizer", display_img)
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
