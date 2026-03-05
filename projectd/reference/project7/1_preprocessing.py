import os
import glob
import random
import lmdb
import pickle
import cv2
from tqdm import tqdm

from blur_ops import init as blur_init, generate_focus_stack, DEVICE, TARGET_H, TARGET_W

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

    # 2. Initialization
    blur_init()
    
    # 3. DB Setup
    if os.path.exists(OUTPUT_IMG_LMDB_PATH):
        print(f"Warning: Output LMDB {OUTPUT_IMG_LMDB_PATH} already exists. It will be appended/overwritten.")
        
    img_env = lmdb.open(OUTPUT_IMG_LMDB_PATH, map_size=MAP_SIZE)
    meta_info = []
    global_counter = 0

    print("Starting generation...")
    
    # 4. Main Generation Loop
    a_list = list(range(10))
    img_txn = img_env.begin(write=True)
    
    try:
        for idx, img_path in enumerate(tqdm(selected_paths)):
            # Background Process
            bg_bgr = cv2.imread(img_path)
            if bg_bgr is None:
                continue
            bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
            
            h, w = bg_gray.shape
            if h > w:
                bg_gray = cv2.rotate(bg_gray, cv2.ROTATE_90_CLOCKWISE)
                h, w = w, h
                
            if h < TARGET_H or w < TARGET_W:
                continue
            
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
                
            bg_gray = cv2.resize(bg_gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            
            # Foreground Process
            fg_gray = get_random_fg_image(all_paths, img_path, TARGET_H, TARGET_W)
            if fg_gray is None:
                continue
                
            # Relative depth offset for background
            c_depth = random.choice([2, 3, 4, 5])
            
            # Generate Focus Stack (only need images, labels, and sharpness)
            blended_imgs, labels, sharpnesses = generate_focus_stack(bg_gray, fg_gray, a_list, c_depth)
            
            # Save to LMDB and Meta
            for i in range(10):
                key_str = f"image_{global_counter:08d}"
                
                img_data = blended_imgs[i]
                img_txn.put(key_str.encode('ascii'), img_data.tobytes())
                
                label_radius = labels[i]
                sharpness_grid = sharpnesses[i]
                meta_info.append((key_str, label_radius, sharpness_grid))
                global_counter += 1
            
            # Periodic Commit to save memory
            if global_counter % 5000 == 0:
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
