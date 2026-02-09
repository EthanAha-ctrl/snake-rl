import os
import lmdb
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import get_hrnet_w18
from train import CoCDataset, LMDB_PATH, META_PATH, DEVICE

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
OUTPUT_LMDB_PATH = os.path.join(DATA_ROOT, "coc_tensor_10x15x20.lmdb")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth")
BATCH_SIZE = 32
NUM_WORKERS = 4
MAP_SIZE = 10000000000 # 10GB, plenty for 2000 tiny tensors

def main():
    print(f"Using device: {DEVICE}")
    dataset = CoCDataset(LMDB_PATH, META_PATH, transform=None)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    model = get_hrnet_w18(num_classes=10, in_channels=1).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Warning: Checkpoint not found! Using random weights.")
    
    model.eval()
    
    if os.path.exists(OUTPUT_LMDB_PATH):
        print(f"Warning: Output LMDB {OUTPUT_LMDB_PATH} already exists. It will be appended/overwritten.")
        
    env = lmdb.open(OUTPUT_LMDB_PATH, map_size=MAP_SIZE)
    
    print("Starting generation...")
    
    count = 0
    with env.begin(write=True) as txn:
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(tqdm(loader)):
                imgs = imgs.to(DEVICE) # [B, 1, 480, 640]
                small_tensors = model(imgs) # [B, 10, 480, 640]
                small_tensors_np = small_tensors.cpu().numpy().astype(np.float32)
                start_global_idx = batch_idx * BATCH_SIZE
                
                for i in range(imgs.size(0)):
                    global_idx = start_global_idx + i
                    meta_item = dataset.meta_info[global_idx]
                    key_str = meta_item[0]
                    tensor_data = small_tensors_np[i]
                    txn.put(key_str.encode('ascii'), tensor_data.tobytes())
                    count += 1
                    
                if count % 1000 == 0:
                     print(f"Processed {count} images...")

    print(f"Done. Saved {count} tensors to {OUTPUT_LMDB_PATH}")
    print(f"Tensor shape: {small_tensors_np[0].shape} (Channels, Height, Width)")

if __name__ == "__main__":
    main()
