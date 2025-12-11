import os
import glob
import random
import cv2
import numpy as np
import lmdb
import pickle
import sys

DATA_ROOT = "project6/data"
OUTPUT_LMDB = os.path.join(DATA_ROOT, "coc_train.lmdb")
OUTPUT_META = os.path.join(DATA_ROOT, "coc_meta.pkl")

TARGET_SIZE = (640, 480)
TOTAL_SOURCE_IMAGES = 2000
CROPS_PER_IMAGE = 5
VARIANTS_PER_CROP = 10 
MAP_SIZE = 1099511627776 

def get_image_paths():
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_paths = []
    for folder_name in ['set1', 'set2', 'set3']:
        folder_path = os.path.join(DATA_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} not found.")
            continue
            
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
    
    print(f"Found {len(image_paths)} total images.")
    return image_paths

def generate_kernel(radius):
    if radius < 0.5:
        return None
    
    scale = 8
    aa_radius = radius * scale
    kernel_size_aa = int(2 * aa_radius) + 1
    if kernel_size_aa % 2 == 0: kernel_size_aa += 1
    
    kernel_aa = np.zeros((kernel_size_aa, kernel_size_aa), dtype=np.float32)
    center = kernel_size_aa // 2
    cv2.circle(kernel_aa, (center, center), int(aa_radius), 1.0, -1)
    
    target_ksize = int(2 * radius) + 1
    if target_ksize % 2 == 0: target_ksize += 1
    
    kernel = cv2.resize(kernel_aa, (target_ksize, target_ksize), interpolation=cv2.INTER_LANCZOS4)
    
    kernel_sum = np.sum(kernel)
    if kernel_sum > 1e-6:
        kernel /= kernel_sum
        
    return kernel

def main():
    all_paths = get_image_paths()
    if len(all_paths) < TOTAL_SOURCE_IMAGES:
        print(f"Warning: Only found {len(all_paths)} images, which is less than target {TOTAL_SOURCE_IMAGES}.")
        selected_paths = all_paths
    else:
        random.seed(42)
        selected_paths = random.sample(all_paths, TOTAL_SOURCE_IMAGES)
    
    print(f"Selected {len(selected_paths)} images for processing.")

    if os.path.exists(OUTPUT_LMDB):
         print(f"Creating new LMDB at {OUTPUT_LMDB}...")
    
    env = lmdb.open(OUTPUT_LMDB, map_size=MAP_SIZE)
    
    print("Pre-calculating kernels...")
    kernels = {}
    for r in range(1, 51):
        kernels[r] = generate_kernel(r)
    
    meta_info = []
    global_counter = 0

    with env.begin(write=True) as txn:
        for idx, img_path in enumerate(selected_paths):
            if idx % 50 == 0:
                print(f"Processing image {idx}/{len(selected_paths)}...")

            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                h, w = img.shape
                if h > w:
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    h, w = w, h
                
                target_crop_w = int(w * 0.5)
                
                if target_crop_w < 640:
                    target_crop_w = 640
                
                if target_crop_w > w:
                    target_crop_w = w
                
                target_crop_h = int(target_crop_w * 2 / 3.0)
                
                if target_crop_h > h:
                    target_crop_h = h
                    target_crop_w = int(target_crop_h * 3 / 2.0)
                    if target_crop_w > w:
                        pass 

                c_w, c_h = target_crop_w, target_crop_h
                
                p1 = ((h - c_h)//2, (w - c_w)//2)
                p2 = (0, 0)
                p3 = (0, w - c_w)
                p4 = (h - c_h, 0)
                p5 = (h - c_h, w - c_w)
                
                crops_coords = [p1, p2, p3, p4, p5]
                
                for (y, x) in crops_coords:
                    patch = img[y : y+c_h, x : x+c_w]
                    
                    patch_resized = cv2.resize(patch, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
                    
                    selected_radii = np.arange(1, 51)
                    
                    for r in selected_radii:
                        k = kernels[r]
                        
                        if r <= 1 or k is None:
                            final_img = patch_resized
                        else:
                            final_img = cv2.filter2D(patch_resized, -1, k)
                        
                        success, encoded_bytes = cv2.imencode('.png', final_img)
                        if not success:
                            continue
                            
                        key_str = f"image_{global_counter:08d}"
                        txn.put(key_str.encode('ascii'), encoded_bytes.tobytes())
                        
                        meta_info.append((key_str, int(r)))
                        
                        global_counter += 1
                        
                        if global_counter % 5000 == 0:
                            txn.commit()
                            txn = env.begin(write=True)
                            print(f"Committed {global_counter} images...")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    env.close()
    
    print(f"Saving metadata to {OUTPUT_META}...")
    with open(OUTPUT_META, 'wb') as f:
        pickle.dump(meta_info, f)
    
    print(f"Done! Total images: {global_counter}")

if __name__ == "__main__":
    main()