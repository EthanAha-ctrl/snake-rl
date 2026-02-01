import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from model import get_hrnet_w18

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(SCRIPT_DIR, "background.png")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_H, TARGET_W = 480, 640

def generate_kernel_tensor(radius):
    """
    Copied from preprocessing.py to ensure exact same blur generation.
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
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found.")
        return

    # --- 1. Load Model ---
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = get_hrnet_w18(num_classes=50, in_channels=1)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Warning: Checkpoint not found, using random weights!")
    
    model.to(DEVICE)
    model.eval()

    # --- 2. Prepare Data (Crop & Blur) ---
    print("Preparing test data (Crop & Blur)...")
    
    # Load Image
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        print("Failed to read image.")
        return
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Rotate if portrait
    h, w = img_gray.shape
    if h > w:
        img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)
        h, w = w, h
    
    # To GPU for processing
    img_tensor = torch.from_numpy(img_gray).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

    # Calculate Crops (Same logic as preprocessing.py)
    target_crop_w = max(640, min(w, int(w * 0.5)))
    target_crop_h = int(target_crop_w * 2 / 3.0)
    if target_crop_h > h:
        target_crop_h = h
        target_crop_w = int(target_crop_h * 1.5)
        if target_crop_w > w: target_crop_w = w 
    
    c_w, c_h = target_crop_w, target_crop_h
    
    # 5 Crops
    p1 = ((h - c_h)//2, (w - c_w)//2)
    p2 = (0, 0)
    p3 = (0, w - c_w)
    p4 = (h - c_h, 0)
    p5 = (h - c_h, w - c_w)
    crops_coords = [p1, p2, p3, p4, p5]
    
    patches_list = []
    for (y, x) in crops_coords:
        crop = img_tensor[:, :, y:y+c_h, x:x+c_w]
        resized = F.interpolate(crop, size=(TARGET_H, TARGET_W), mode='bicubic', align_corners=False)
        patches_list.append(resized)
    
    batch_patches = torch.cat(patches_list, dim=0) # [5, 1, 480, 640]

    # Pre-calculate Kernels
    print("Generating kernels...")
    kernels = {}
    for r in range(1, 51):
        kernels[r] = generate_kernel_tensor(r)

    # Generate all variants
    # Store as list of dictionaries or tuples
    test_samples = [] # List of {"image": cv2_img, "label": int}

    print("Generating blurred variants...")
    for r in range(1, 51):
        k_tensor = kernels[r]
        
        if r <= 1 or k_tensor is None:
            out_tensor = batch_patches
        else:
            k_h, k_w = k_tensor.shape[2:]
            pad_h, pad_w = k_h // 2, k_w // 2
            padded = F.pad(batch_patches, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            out_tensor = F.conv2d(padded, k_tensor, padding=0)
        
        # To CPU for visualization storage
        # [5, 1, H, W] -> [5, H, W]
        out_tensor = out_tensor.clamp(0, 255).byte()
        cpu_imgs = out_tensor.squeeze(1).cpu().numpy()
        
        for i in range(5):
            test_samples.append({
                "image": cpu_imgs[i],
                "label": r,
                "crop_id": i
            })
    
    # Sort by Crop then Radius for smoother viewing, or keep as is (Radius varies fast)
    # preprocessing.py loop was: Radius outer, then Crop. So list is: R1[C1..C5], R2[C1..C5]...
    # Visualize.py expects: Crop 1 [R1..R50], Crop 2 [R1..R50].
    # Let's sort to match visualize.py's preferred viewing order (Crop varies slowly)
    test_samples.sort(key=lambda x: (x["crop_id"], x["label"]))

    print(f"Generated {len(test_samples)} test samples.")
    print("Starting Inference & Visualization...")
    print("Press 'q' to quit, 'b' for back, any other key for next.")

    idx = 0
    total = len(test_samples)
    
    while True:
        if idx < 0: idx = 0
        if idx >= total: idx = 0
        
        sample = test_samples[idx]
        img_vis = sample["image"]
        label = sample["label"]
        crop_id = sample["crop_id"]
        
        # Inference
        # Prepare input: [H, W] -> float32 0-1 -> [1, 1, H, W]
        input_tensor = img_vis.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            logits_avg = output.mean(dim=(2, 3))
            pred_idx = logits_avg.argmax(1).item()
            pred_radius = pred_idx + 1

        # Calculate error
        diff = abs(pred_radius - label)
        status = "OK" if diff == 0 else f"Err: {diff}"
        color_str = ""
        if diff <= 1: color_str = "(Good)"
        elif diff > 5: color_str = "(Bad)"

        title = f"Crop:{crop_id} | GT:{label} | Pred:{pred_radius} {status} {color_str} | {idx+1}/{total}"
        
        cv2.imshow("Single Image Eval", img_vis)
        cv2.setWindowTitle("Single Image Eval", title)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            idx -= 1
        else:
            idx += 1
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
