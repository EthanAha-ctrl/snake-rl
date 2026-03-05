import os
import glob
import random
import lmdb
import pickle
import cv2
import time
from tqdm import tqdm

from blur_ops import init as blur_init, core_blur, DEVICE, TARGET_H, TARGET_W

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
VAL_DIR = os.path.join(DATA_ROOT, "val2017")

OUTPUT_IMG_LMDB_PATH = os.path.join(DATA_ROOT, "coc_img_10x480x640.lmdb")
OUTPUT_META_PATH = os.path.join(DATA_ROOT, "coc_meta.pkl")

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

def process_image_wrapper(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return []

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    h, w = img_gray.shape
    if h > w:
        img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)
        h, w = w, h

    if h < TARGET_H or w < TARGET_W:
        return []

    target_aspect = TARGET_W / TARGET_H
    current_aspect = w / h
    
    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        start_x = (w - new_w) // 2
        img_gray = img_gray[:, start_x : start_x + new_w]
    else:
        new_h = int(w / target_aspect)
        start_y = (h - new_h) // 2
        img_gray = img_gray[start_y : start_y + new_h, :]
        
    img_gray = cv2.resize(img_gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    outputs = []
    
    a_list = [0] * 10
    b_list = list(range(0, 10))
    
    blurred_batch, sharpness_batch = core_blur(img_gray, a_list, b_list)
    
    for i, r in enumerate(b_list):
        blurred_img = blurred_batch[i]
        sharpness_grid = sharpness_batch[i]
        outputs.append((blurred_img, r, sharpness_grid))
            
    return outputs

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
        
    print(f"Selected {len(selected_paths)} images for processing.")

    # 2. Initialization
    blur_init()
    
    # 3. DB Setup
    if os.path.exists(OUTPUT_IMG_LMDB_PATH):
        print(f"Warning: Output LMDB {OUTPUT_IMG_LMDB_PATH} already exists. It will be appended/overwritten.")
        
    img_env = lmdb.open(OUTPUT_IMG_LMDB_PATH, map_size=MAP_SIZE)
    meta_info = []
    global_counter = 0

    print("Starting generation...")
    img_txn = img_env.begin(write=True)
    
    try:
        for idx, img_path in enumerate(tqdm(selected_paths)):
            try:
                processed_results = process_image_wrapper(img_path)
                
                for (final_img, r, sharpness_grid) in processed_results:
                    success, encoded_bytes = cv2.imencode('.png', final_img)
                    if success:
                        key_str = f"image_{global_counter:08d}"
                        img_txn.put(key_str.encode('ascii'), encoded_bytes.tobytes())
                        meta_info.append((key_str, int(r), sharpness_grid))
                        global_counter += 1
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
                
            # Periodic Commit to save memory
            if global_counter > 0 and global_counter % 5000 == 0:
                img_txn.commit()
                img_txn = img_env.begin(write=True)
                    
        img_txn.commit()
    except Exception as e:
        img_txn.abort()
        print(f"Aborted due to error: {e}")
        raise
    finally:
        img_env.close()

    print(f"Done. Saved {global_counter} images to {OUTPUT_IMG_LMDB_PATH}")
    print(f"Saving metadata to {OUTPUT_META_PATH}...")
    with open(OUTPUT_META_PATH, 'wb') as f:
        pickle.dump(meta_info, f)

if __name__ == "__main__":
    main()
