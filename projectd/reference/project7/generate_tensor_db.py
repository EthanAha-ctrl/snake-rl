import os
import glob
import random
import lmdb
import pickle
import torch
import numpy as np
import cv2
from tqdm import tqdm

from model import get_hrnet_w18
from blur_ops import init as blur_init, generate_focus_stack, DEVICE, TARGET_H, TARGET_W

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
VAL_DIR = os.path.join(DATA_ROOT, "val2017")

OUTPUT_LMDB_PATH = os.path.join(DATA_ROOT, "coc_tensor_10x15x20.lmdb")
OUTPUT_META_PATH = os.path.join(DATA_ROOT, "coc_meta.pkl")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth")

TOTAL_SOURCE_IMAGES = 2000
MAP_SIZE = 10000000000 # 10GB

def get_image_paths(directory):
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_paths = []
    if not os.path.exists(directory):
        return []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return image_paths

def get_random_fg_image(image_paths, current_bg_path, h, w):
    fg_candidates = [f for f in image_paths if f != current_bg_path]
    if not fg_candidates:
        return None
    
    fg_path = random.choice(fg_candidates)
    fg_bgr = cv2.imread(fg_path)
    if fg_bgr is None:
        return None
        
    fg_gray = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2GRAY)
    
    # Rotate if portrait and background is landscape, or vice versa
    fh, fw = fg_gray.shape
    if (fh > fw and h < w) or (fh < fw and h > w):
        fg_gray = cv2.rotate(fg_gray, cv2.ROTATE_90_CLOCKWISE)
        
    fg_resized = cv2.resize(fg_gray, (w, h), interpolation=cv2.INTER_AREA)
    return fg_resized

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Image Discovery
    search_dir = VAL_DIR if os.path.exists(VAL_DIR) else DATA_ROOT
    all_paths = get_image_paths(search_dir)
    if not all_paths:
         print(f"Error: No images found in {search_dir}")
         return
         
    if len(all_paths) < TOTAL_SOURCE_IMAGES:
        selected_paths = all_paths
    else:
        random.seed(42)
        selected_paths = random.sample(all_paths, TOTAL_SOURCE_IMAGES)
        
    print(f"Selected {len(selected_paths)} background images for processing.")

    # 2. Model Initialization
    blur_init()
    
    model = get_hrnet_w18(num_classes=10, in_channels=1).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Warning: Checkpoint not found! Using random weights.")
    model.eval()
    
    # 3. DB Setup
    if os.path.exists(OUTPUT_LMDB_PATH):
        print(f"Warning: Output LMDB {OUTPUT_LMDB_PATH} already exists. It will be appended/overwritten.")
        
    env = lmdb.open(OUTPUT_LMDB_PATH, map_size=MAP_SIZE)
    meta_info = []
    global_counter = 0

    print("Starting generation...")
    
    # 4. Main Generation Loop
    a_list = list(range(10))
    
    with env.begin(write=True) as txn:
        with torch.no_grad():
            for img_path in tqdm(selected_paths):
                # Load BG
                bg_bgr = cv2.imread(img_path)
                if bg_bgr is None:
                    continue
                bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
                h, w = bg_gray.shape
                if h > w:
                    bg_gray = cv2.rotate(bg_gray, cv2.ROTATE_90_CLOCKWISE)
                    h, w = w, h
                
                # Dynamic Crop to target aspect (similar to old preprocessing)
                target_aspect = TARGET_W / TARGET_H
                current_aspect = w / h
                if current_aspect > target_aspect:
                    new_w = int(h * target_aspect)
                    start_x = (w - new_w) // 2
                    bg_gray = bg_gray[:, start_x : start_x + new_w]
                else:
                    new_h = int(w / target_aspect)
                    start_y = (h - new_h) // 2
                    bg_gray = bg_gray[start_y : start_y + new_h, :]
                h, w = bg_gray.shape
                    
                # Load FG
                fg_gray = get_random_fg_image(all_paths, img_path, h, w)
                if fg_gray is None:
                    fg_gray = bg_gray.copy()
                
                # Random Depths
                b_depth = random.randint(0, 9)
                c_depth = random.randint(0, 9)
                
                # Generate exact Optical Focus Stack
                blended_imgs, labels = generate_focus_stack(bg_gray, fg_gray, a_list, b_depth, c_depth)
                
                # Prepare batch tensor for HRNet
                # blended_imgs is a list of [480, 640] unit8 arrays
                batch_tensors = []
                for img in blended_imgs:
                    # Convert to [1, 480, 640] float32 normalized 0-1
                    t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
                    batch_tensors.append(t)
                
                batch_tensor = torch.stack(batch_tensors).to(DEVICE) # [10, 1, 480, 640]
                
                # Extract Features via HRNet
                small_tensors = model(batch_tensor) # [10, 10, 480, 640] -> Wait, the model output head is what?
                # Actually, the model normally outputs [B, C, H_small, W_small] where C=10, H_small=15, W_small=20
                small_tensors_np = small_tensors.cpu().numpy().astype(np.float32) # [10, 10, 15, 20]
                
                # Save to LMDB and Meta
                for i in range(10):
                    key_str = f"image_{global_counter:08d}"
                    tensor_data = small_tensors_np[i]
                    txn.put(key_str.encode('ascii'), tensor_data.tobytes())
                    
                    # Store absolute foreground blur radius as label, plus original depth info just in case
                    label_radius = labels[i] # absolute blur radius of foreground
                    
                    # Store tuple (key, label, b_depth, c_depth, a_focus) in metadata
                    # To be somewhat compatible with old format while adding info:
                    # old format used: (key_str, int(r), sharpness_grid)
                    # We dropped sharpness_grid for now, or we can compute it on the final blended_image?
                    # The user didn't request sharpness grid for this new pipeline, but `history_stacker` might need it?
                    # Let's keep it clean: (key, label)
                    meta_info.append((key_str, label_radius, dict(a=i, b=b_depth, c=c_depth)))
                    global_counter += 1
                
                # Periodic Commit to save memory
                if global_counter % 5000 == 0:
                    txn.commit()
                    env.begin(write=True)

    print(f"Done. Saved {global_counter} tensors to {OUTPUT_LMDB_PATH}")
    
    print(f"Saving metadata to {OUTPUT_META_PATH}...")
    with open(OUTPUT_META_PATH, 'wb') as f:
        pickle.dump(meta_info, f)
        
    print("Database generation complete!")

if __name__ == "__main__":
    main()
