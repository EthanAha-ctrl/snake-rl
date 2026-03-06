import os
import glob
import random
import lmdb
import pickle
import torch
import numpy as np
import cv2
from tqdm import tqdm

from model import MiniHRNetMIL
from blur_ops import init as blur_init, generate_focus_stack, DEVICE, TARGET_H, TARGET_W

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
VAL_DIR = os.path.join(DATA_ROOT, "val2017")

OUTPUT_LMDB_PATH = os.path.join(DATA_ROOT, "coc_tensor_10x15x20.lmdb")
OUTPUT_IMG_LMDB_PATH = os.path.join(DATA_ROOT, "coc_foreground_background_10x480x640.lmdb")
OUTPUT_META_PATH = os.path.join(DATA_ROOT, "coc_meta_foreground_background.pkl")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "best_model_patch.pth")

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
    
    model = MiniHRNetMIL(num_classes=10).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Warning: Checkpoint not found! Using random weights.")
    model.eval()
    
    if os.path.exists(OUTPUT_LMDB_PATH):
        print(f"Warning: Output LMDB {OUTPUT_LMDB_PATH} already exists. It will be appended/overwritten.")
    if os.path.exists(OUTPUT_IMG_LMDB_PATH):
        print(f"Warning: Output Image LMDB {OUTPUT_IMG_LMDB_PATH} already exists. It will be appended/overwritten.")
        
    env = lmdb.open(OUTPUT_LMDB_PATH, map_size=MAP_SIZE)
    img_env = lmdb.open(OUTPUT_IMG_LMDB_PATH, map_size=MAP_SIZE)
    meta_info = []
    global_counter = 0

    print("Starting generation...")
    
    # 4. Main Generation Loop
    a_list = list(range(10))
    
    txn = env.begin(write=True)
    img_txn = img_env.begin(write=True)
    try:
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
                
                # Dynamic Crop to target aspect
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
                
                # Resize strictly to TARGET_W and TARGET_H before generation
                bg_gray = cv2.resize(bg_gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
                fg_gray = cv2.resize(fg_gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

                # Relative depth offset for background
                c_depth = random.choice([2, 3, 4, 5])
                
                # Generate exact Optical Focus Stack
                blended_imgs, labels, sharpnesses = generate_focus_stack(bg_gray, fg_gray, a_list, c_depth)
                
                # Prepare batch tensor for MILPatchCNN
                batch_tensors = []
                for img in blended_imgs:
                    t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
                    batch_tensors.append(t)
                
                batch_tensor = torch.stack(batch_tensors).to(DEVICE) # [10, 1, 480, 640]
                
                # Extract Features via MILPatchCNN
                # inference returns logits with shape [10, 10, 15, 20]
                spatial_logits = model(batch_tensor) 
                spatial_logits_np = spatial_logits.cpu().numpy().astype(np.float32) 
                
                # Save to LMDB and Meta
                for i in range(20):
                    key_str = f"image_{global_counter:08d}"
                    
                    # 1. Save Tensor
                    tensor_data = spatial_logits_np[i] # [10, 15, 20]
                    txn.put(key_str.encode('ascii'), tensor_data.tobytes())
                    
                    # 2. Save Image
                    success, encoded_bytes = cv2.imencode('.png', blended_imgs[i])
                    if success:
                         img_txn.put(key_str.encode('ascii'), encoded_bytes.tobytes())
                    
                    label_data = labels[i] # (sign, radius)
                    sharpness_grid = sharpnesses[i]
                    meta_info.append((key_str, label_data, sharpness_grid))
                    global_counter += 1
                
                # Periodic Commit to save memory
                if global_counter > 0 and global_counter % 5000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    img_txn.commit()
                    img_txn = img_env.begin(write=True)
                    
        txn.commit()
        img_txn.commit()
    except Exception as e:
        txn.abort()
        img_txn.abort()
        print(f"Aborted due to error: {e}")
        raise
    finally:
        env.close()
        img_env.close()

    print(f"Done. Saved {global_counter} tensors to {OUTPUT_LMDB_PATH}")
    
    print(f"Saving metadata to {OUTPUT_META_PATH}...")
    with open(OUTPUT_META_PATH, 'wb') as f:
        pickle.dump(meta_info, f)
        
    print("Database generation complete!")

if __name__ == "__main__":
    main()
