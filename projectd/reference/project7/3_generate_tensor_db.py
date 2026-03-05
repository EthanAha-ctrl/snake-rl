import os
import lmdb
import pickle
import torch
import numpy as np
import cv2
from tqdm import tqdm

from model import MILPatchCNN
from blur_ops import DEVICE

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")

INPUT_IMG_LMDB_PATH = os.path.join(DATA_ROOT, "coc_img_10x480x640.lmdb")
OUTPUT_LMDB_PATH = os.path.join(DATA_ROOT, "coc_tensor_10x15x20.lmdb")
META_PATH = os.path.join(DATA_ROOT, "coc_meta.pkl")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "best_model_patch.pth")

MAP_SIZE = 10000000000 # 10GB

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Dependency Checks Setup
    if not os.path.exists(INPUT_IMG_LMDB_PATH) or not os.path.exists(META_PATH):
        print(f"Error: Required datasets not found. Please run 1_preprocessing.py first.")
        return
        
    print(f"Loading metadata from {META_PATH}...")
    with open(META_PATH, 'rb') as f:
        meta_info = pickle.load(f)
        
    img_env = lmdb.open(INPUT_IMG_LMDB_PATH, readonly=True, lock=False)
    
    # 2. Model Initialization
    model = MILPatchCNN(num_classes=10).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Warning: Checkpoint not found! Using random weights.")
    model.eval()
    
    # 3. Output DB Setup
    if os.path.exists(OUTPUT_LMDB_PATH):
        print(f"Warning: Output LMDB {OUTPUT_LMDB_PATH} already exists. It will be appended/overwritten.")
        
    tensor_env = lmdb.open(OUTPUT_LMDB_PATH, map_size=MAP_SIZE)
    global_counter = 0

    print("Starting Tensor DB generation from preprocessed images...")
    
    # We process in batches of 10 (one full scene Focus Stack)
    total_images = len(meta_info)
    assert total_images % 10 == 0, "Metadata length should be a multiple of 10"
    total_scenes = total_images // 10

    tensor_txn = tensor_env.begin(write=True)
    
    try:
        with torch.no_grad():
            for scene_idx in tqdm(range(total_scenes)):
                scene_start = scene_idx * 10
                
                # Buffer for the scene's images
                scene_imgs = []
                keys = []
                
                with img_env.begin() as img_txn:
                    for i in range(10):
                        idx = scene_start + i
                        key_str, _, _ = meta_info[idx]
                        keys.append(key_str)
                        
                        img_bytes = img_txn.get(key_str.encode('ascii'))
                        img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape(480, 640)
                        
                        # Normalize to 0-1 for CNN
                        img_normalized = img_array.astype(np.float32) / 255.0
                        scene_imgs.append(img_normalized)
                
                # Shape: [10, 1, 480, 640]
                img_tensor = torch.from_numpy(np.array(scene_imgs)).unsqueeze(1).to(DEVICE)
                
                # Inference! Returns [10, 10, 15, 20]
                # B=10 images in the scene, 10 classes, 15H x 20W spatial patches
                spatial_logits = model(img_tensor)
                
                # Move to CPU numpy
                spatial_logits_np = spatial_logits.cpu().numpy()
                
                # Save Tensors to LMDB
                for i in range(10):
                    tensor_data = spatial_logits_np[i] # [10, 15, 20]
                    tensor_txn.put(keys[i].encode('ascii'), tensor_data.tobytes())
                    global_counter += 1
                
                # Periodic Commit to save memory
                if global_counter % 5000 == 0:
                    tensor_txn.commit()
                    tensor_txn = tensor_env.begin(write=True)
                    
        tensor_txn.commit()
        
    except Exception as e:
        tensor_txn.abort()
        print(f"Aborted due to error: {e}")
        raise
        
    finally:
        tensor_env.close()
        img_env.close()

    print(f"Done. Saved {global_counter} scene tensors to {OUTPUT_LMDB_PATH}")

if __name__ == "__main__":
    main()
