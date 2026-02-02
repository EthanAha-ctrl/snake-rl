import os
import glob
import random
import cv2
import lmdb
import pickle
import time
from blur_ops import core_blur, init_gpu_kernels, DEVICE, TARGET_H, TARGET_W

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
OUTPUT_LMDB = os.path.join(DATA_ROOT, "coc_train.lmdb")
OUTPUT_META = os.path.join(DATA_ROOT, "coc_meta.pkl")

TOTAL_SOURCE_IMAGES = 2000
MAP_SIZE = 10000000000

print(f"Using device: {DEVICE}")

def get_image_paths():
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(DATA_ROOT, "**", ext), recursive=True))
    
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
        raise ValueError(f"Image too small after rotation: {w}x{h} < {TARGET_W}x{TARGET_H}")

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
    
    outputs = []
    
    a_list = [0] * 50
    b_list = list(range(1, 51))
    
    blurred_batch, sharpness_batch = core_blur(img_gray, a_list, b_list)
    
    for i, r in enumerate(b_list):
        blurred_img = blurred_batch[i]
        sharpness_grid = sharpness_batch[i]
        outputs.append((blurred_img, r, sharpness_grid))
            
    return outputs


def main():
    all_paths = get_image_paths()
    if len(all_paths) < TOTAL_SOURCE_IMAGES:
        selected_paths = all_paths
    else:
        random.seed(42)
        selected_paths = random.sample(all_paths, TOTAL_SOURCE_IMAGES)
    
    print(f"Selected {len(selected_paths)} images.")

    if os.path.exists(OUTPUT_LMDB):
         print(f"Creating new LMDB at {OUTPUT_LMDB}...")
    
    env = lmdb.open(OUTPUT_LMDB, map_size=MAP_SIZE)
    
    init()
    
    meta_info = []
    global_counter = 0

    start_time = time.time()
    
    txn = env.begin(write=True)
    try:
        for idx, img_path in enumerate(selected_paths):
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                if idx > 0:
                    avg_time = elapsed / idx
                    remaining = (len(selected_paths) - idx) * avg_time
                    rem_h = int(remaining // 3600)
                    rem_m = int((remaining % 3600) // 60)
                    print(f"Processing image {idx}/{len(selected_paths)}... ETA: {rem_h}h {rem_m}m")
                else:
                    print(f"Processing image {idx}/{len(selected_paths)}...")

            try:
                processed_results = process_image_wrapper(img_path)
                
                for (final_img, r, sharpness_grid) in processed_results:
                    success, encoded_bytes = cv2.imencode('.png', final_img)
                    if success:
                        key_str = f"image_{global_counter:08d}"
                        txn.put(key_str.encode('ascii'), encoded_bytes.tobytes())
                        meta_info.append((key_str, int(r), sharpness_grid))
                        global_counter += 1
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
                
            if global_counter % 5000 == 0:
                txn.commit()
                txn = env.begin(write=True)
                
        txn.commit()
        
    except Exception as e:
        txn.abort()
        print(f"Aborted due to error: {e}")
        raise
    finally:
        env.close()
    
    print(f"Saving metadata to {OUTPUT_META}...")
    with open(OUTPUT_META, 'wb') as f:
        pickle.dump(meta_info, f)
    
    print(f"Done! Total images: {global_counter}")

if __name__ == "__main__":
    main()