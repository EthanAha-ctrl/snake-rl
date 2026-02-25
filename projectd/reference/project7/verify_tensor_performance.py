import lmdb
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import os
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
OUTPUT_LMDB_PATH = os.path.join(DATA_ROOT, "coc_tensor_10x15x20.lmdb")
META_PATH = os.path.join(DATA_ROOT, "coc_meta.pkl")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    if not os.path.exists(OUTPUT_LMDB_PATH):
        print(f"Error: Database not found at {OUTPUT_LMDB_PATH}")
        return
        
    if not os.path.exists(META_PATH):
        print(f"Error: Metadata not found at {META_PATH}")
        return

    print("Loading metadata...")
    with open(META_PATH, 'rb') as f:
        meta_info = pickle.load(f)
    print(f"Loaded {len(meta_info)} metadata entries.")
    
    # Create key-to-label map for fast lookup
    # meta_info: [(key, label), ...] or [(key, label, extra), ...]
    key_to_label = {}
    for item in meta_info:
        if len(item) == 3:
            key, label, _ = item
        else:
            key, label = item
        key_to_label[key] = int(label)

    # Per-label error collection
    label_errors = {i: [] for i in range(10)}

    print(f"Opening LMDB: {OUTPUT_LMDB_PATH}")
    env = lmdb.open(OUTPUT_LMDB_PATH, readonly=True, lock=False)
    
    # Class values for expectation: [0, 1, 2, ..., 9]
    class_values = torch.arange(0, 10, device=DEVICE, dtype=torch.float32).unsqueeze(0)

    total_error = 0.0
    count = 0

    print("Verifying predictions...")
    with env.begin() as txn:
        cursor = txn.cursor()
        
        for key_bytes, value_bytes in tqdm.tqdm(cursor, total=len(meta_info)):
            key_str = key_bytes.decode('ascii')
            
            if key_str not in key_to_label:
                continue
                
            label = key_to_label[key_str]
            
            # Get Tensor & Compute
            tensor_np = np.frombuffer(value_bytes, dtype=np.float32).reshape(10, 15, 20)
            input_tensor = torch.from_numpy(tensor_np).unsqueeze(0).to(DEVICE)
            
            logits_avg = input_tensor.mean(dim=(2, 3))
            # Top-3 Soft Expectation
            # 1. Get Probabilities
            probs = torch.softmax(logits_avg, dim=1) # [1, 10]
            
            # 2. Mask non-Top-3 to 0
            k = 2
            topk_vals, topk_indices = torch.topk(probs, k, dim=1)
            
            mask = torch.zeros_like(probs)
            mask.scatter_(1, topk_indices, 1.0)
            
            masked_probs = probs * mask
            
            # 3. Re-normalize
            masked_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-9)
            
            # 4. Expectation
            expected_radius = (masked_probs * class_values).sum(dim=1).item()
            # Error
            diff = abs(expected_radius - label)
            label_errors[label].append(diff)
            
            total_error += diff
            count += 1
            
    print(f"\n[Validation Summary | Total Samples: {count}]")
    print(f"Global MAE: {total_error / count:.4f}")
    
    print("\n[Per-Label Error Distribution]")
    print(f"{'Label':<6} | {'Count':<6} | {'MAE':<6} | {'Error Histogram (Buckets: 0-0.1, 0.1-0.5, 0.5-1.0, >1.0)'}")
    print("-" * 80)
    
    for label in range(10):
        errors = label_errors[label]
        if not errors:
            print(f"{label:<6} | {0:<6} | {'N/A':<6} |")
            continue
            
        mae = sum(errors) / len(errors)
        
        # Simple Histogram Buckets
        n_perfect = sum(1 for e in errors if e < 0.1)
        n_good = sum(1 for e in errors if 0.1 <= e < 0.5)
        n_ok = sum(1 for e in errors if 0.5 <= e < 1.0)
        n_bad = sum(1 for e in errors if e >= 1.0)
        
        total = len(errors)
        
        # Stats
        p_perfect = (n_perfect / total) * 100
        p_good = (n_good / total) * 100
        p_ok = (n_ok / total) * 100
        p_bad = (n_bad / total) * 100
        
        hist_str = f"<=0.1: {n_perfect:<3} ({p_perfect:4.1f}%) | <=0.5: {n_good:<3} ({p_good:4.1f}%) | <=1.0: {n_ok:<3} ({p_ok:4.1f}%) | >1.0: {n_bad:<3} ({p_bad:4.1f}%)"
        print(f"{label:<6} | {total:<6} | {mae:.4f} | {hist_str}")

if __name__ == "__main__":
    main()
