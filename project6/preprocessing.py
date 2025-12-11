import os
import glob
import random
import cv2
import numpy as np
import lmdb
import pickle
import sys
import torch
import torch.nn.functional as F

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
OUTPUT_LMDB = os.path.join(DATA_ROOT, "coc_train.lmdb")
OUTPUT_META = os.path.join(DATA_ROOT, "coc_meta.pkl")

TARGET_H, TARGET_W = 480, 640
TOTAL_SOURCE_IMAGES = 2000
CROPS_PER_IMAGE = 5
# VARIANTS_PER_CROP = 50 (Implicitly 50 because we iterate 1..50)

MAP_SIZE = 1099511627776 

# Check CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def get_image_paths():
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_paths = []
    for folder_name in ['set1', 'set2', 'set3']:
        folder_path = os.path.join(DATA_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
    
    return image_paths

def generate_kernel_tensor(radius):
    """
    生成并返回 GPU 上的 Kernel Tensor
    Shape: [1, 1, K, K]
    """
    if radius < 0.5:
        return None 
    
    # 1. High Res Calculation (Anti-aliasing)
    scale = 8
    aa_radius = radius * scale
    kernel_size_aa = int(2 * aa_radius) + 1
    if kernel_size_aa % 2 == 0: kernel_size_aa += 1
    
    kernel_aa = np.zeros((kernel_size_aa, kernel_size_aa), dtype=np.float32)
    center = kernel_size_aa // 2
    cv2.circle(kernel_aa, (center, center), int(aa_radius), 1.0, -1)
    
    # 2. Downscale to Target Kernel Size (using OpenCV Lanczos CPU side first)
    target_ksize = int(2 * radius) + 1
    if target_ksize % 2 == 0: target_ksize += 1
    
    kernel = cv2.resize(kernel_aa, (target_ksize, target_ksize), interpolation=cv2.INTER_LANCZOS4)
    
    # 3. Normalize
    kernel_sum = np.sum(kernel)
    if kernel_sum > 1e-6:
        kernel /= kernel_sum
        
    # Convert to Tensor [OutCh, InCh, H, W] -> [1, 1, H, W]
    k_tensor = torch.from_numpy(kernel).to(DEVICE).unsqueeze(0).unsqueeze(0)
    return k_tensor

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
    
    # Pre-calculate Kernels on GPU
    print("Pre-calculating kernels on GPU...")
    # Dictionary: radius -> tensor
    kernels = {}
    for r in range(1, 51):
        kernels[r] = generate_kernel_tensor(r)
    
    meta_info = []
    global_counter = 0

    txn = env.begin(write=True)
    try:
        for idx, img_path in enumerate(selected_paths):
            if idx % 10 == 0:
                print(f"Processing image {idx}/{len(selected_paths)}...")

            try:
                # 1. CPU Load & Decode
                img_bgr = cv2.imread(img_path)
                if img_bgr is None: continue
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                
                # Retrieve Shape
                h, w = img_gray.shape
                # Rotate Portrait
                if h > w:
                    img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)
                    h, w = w, h
                
                # 2. Upload to GPU
                # [H, W] -> [1, 1, H, W] (Float for processing)
                img_tensor = torch.from_numpy(img_gray).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

                # 3. Determine Crops (CPU logic for coordinates is fine)
                target_crop_w = max(640, min(w, int(w * 0.5)))
                target_crop_h = int(target_crop_w * 2 / 3.0)
                if target_crop_h > h:
                    target_crop_h = h
                    target_crop_w = int(target_crop_h * 1.5)
                    if target_crop_w > w: target_crop_w = w # fallback
                
                c_w, c_h = target_crop_w, target_crop_h
                
                # Define 5 Crop Coordinates
                p1 = ((h - c_h)//2, (w - c_w)//2)
                p2 = (0, 0)
                p3 = (0, w - c_w)
                p4 = (h - c_h, 0)
                p5 = (h - c_h, w - c_w)
                crops_coords = [p1, p2, p3, p4, p5]
                
                # 4. Crop & Resize on GPU
                # We will collect 5 resized patches
                patches_list = []
                for (y, x) in crops_coords:
                    # Slicing tensor on GPU [1, 1, H, W]
                    crop = img_tensor[:, :, y:y+c_h, x:x+c_w]
                    # Resize (Bilinear or Bicubic). Lanczos not natively supported in F.interpolate
                    # 'bicubic' is close enough and very fast on GPU
                    resized = F.interpolate(crop, size=(TARGET_H, TARGET_W), mode='bicubic', align_corners=False)
                    patches_list.append(resized) # Each is [1, 1, 480, 640]
                
                # Stack to [5, 1, 480, 640]
                batch_patches = torch.cat(patches_list, dim=0)

                # 5. Apply CoCs (1 to 50) on GPU
                # Loop through 1..50.
                
                for r in range(1, 51):
                    k_tensor = kernels[r]
                    
                    if r <= 1 or k_tensor is None:
                        # No blur
                        out_tensor = batch_patches
                    else:
                        # Padding logic: Use Reflection Padding to avoid black borders
                        k_h, k_w = k_tensor.shape[2:]
                        pad_h = k_h // 2
                        pad_w = k_w // 2
                        
                        # Pad first, then conv without padding
                        # F.pad expects (left, right, top, bottom)
                        padded_batch = F.pad(batch_patches, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
                        
                        out_tensor = F.conv2d(padded_batch, k_tensor, padding=0)
                    
                    # 6. Download to CPU [5, 1, H, W] -> Byte Numpy
                    # Clamp to 0-255 first
                    out_tensor = out_tensor.clamp(0, 255).byte()
                    # To CPU: [5, 480, 640] (squeeze channel)
                    cpu_imgs = out_tensor.squeeze(1).cpu().numpy()
                    
                    # 7. Encode & Write (CPU)
                    # Iterate the 5 crops
                    for i in range(5):
                        final_img = cpu_imgs[i] # [480, 640]
                        
                        success, encoded_bytes = cv2.imencode('.png', final_img)
                        if success:
                            key_str = f"image_{global_counter:08d}"
                            txn.put(key_str.encode('ascii'), encoded_bytes.tobytes())
                            meta_info.append((key_str, int(r)))
                            global_counter += 1
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
                
            # Commit periodically
            if global_counter % 5000 == 0:
                txn.commit()
                txn = env.begin(write=True)
                
        # Final commit
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