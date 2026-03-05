import os
import lmdb
import pickle
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_ROOT = "data"
LMDB_PATH = os.path.join(DATA_ROOT, "coc_tensor_10x15x20.lmdb")
IMG_LMDB_PATH = os.path.join(DATA_ROOT, "coc_foreground_background_10x480x640.lmdb")
META_PATH = os.path.join(DATA_ROOT, "coc_meta_foreground_background.pkl")

def visualize_dataset():
    if not os.path.exists(LMDB_PATH) or not os.path.exists(META_PATH):
        print("Database not found. Please run generate_tensor_db.py first.")
        return

    print(f"Loading metadata from {META_PATH}...")
    with open(META_PATH, 'rb') as f:
        meta_info = pickle.load(f)

    print(f"Opening LMDB from {LMDB_PATH}...")
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
    img_env = lmdb.open(IMG_LMDB_PATH, readonly=True, lock=False)
    
    total_images = len(meta_info)
    
    cv2.namedWindow('Dataset Visualizer', cv2.WINDOW_AUTOSIZE)
    
    current_idx = 0
    while current_idx < total_images:
        
        # Calculate sharpness scale for the CURRENT scene (group of 10) dynamically
        scene_start_idx = (current_idx // 10) * 10
        max_sharpness = 1.0
        for i in range(10):
            if scene_start_idx + i < total_images:
                _, _, sharpness_grid = meta_info[scene_start_idx + i]
                max_sharpness = max(max_sharpness, np.max(sharpness_grid))
        sharpness_scale = max_sharpness / 2.0
        
        # Get data
        key_str, label, spatial_sharpness = meta_info[current_idx]
        
        with env.begin() as txn:
            tensor_bytes = txn.get(key_str.encode('ascii'))
            t_mix = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(10, 15, 20)
            
        with img_env.begin() as txn:
            img_bytes = txn.get(key_str.encode('ascii'))
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            orig_gray = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if orig_gray is None:
                continue
            orig_bgr = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)
            
        scene_idx = current_idx // 10
        step_idx = current_idx % 10
            
        # 2. Pixel-wise Expected Radius (User Logic)
        tensor = torch.from_numpy(t_mix) # [10, 15, 20]
        probs = torch.softmax(tensor, dim=0)
        
        k = 2
        topk_vals, topk_indices = torch.topk(probs, k, dim=0)
        mask = torch.zeros_like(probs)
        mask.scatter_(0, topk_indices, 1.0)
        
        masked_probs = probs * mask
        masked_probs = masked_probs / (masked_probs.sum(dim=0, keepdim=True) + 1e-9)
        
        class_values = torch.arange(0, 10, dtype=torch.float32).view(10, 1, 1)
        expected_radius = (masked_probs * class_values).sum(dim=0).numpy() # [15, 20]
        
        # 3. Construct Separate Grayscale Images
        
        # map expected_radius 0..9 -> grayscale 0..255
        radius_gray = np.clip((expected_radius / 9.0) * 255.0, 0, 255).astype(np.uint8)
        
        # map sharpness 0..max_sharpness -> grayscale 0..255
        v_channel = np.clip((spatial_sharpness / (max_sharpness + 1e-5)) * 255.0, 0, 255).astype(np.uint8)
        
        # Convert Grayscales to BGR (for OpenCV display with colors/text)
        radius_bgr = cv2.cvtColor(radius_gray, cv2.COLOR_GRAY2BGR)
        sharpness_bgr = cv2.cvtColor(v_channel, cv2.COLOR_GRAY2BGR)
        
        # Apply colormaps for better visibility (optional but recommended for grayscale data)
        radius_bgr = cv2.applyColorMap(radius_bgr, cv2.COLORMAP_JET)
        sharpness_bgr = cv2.applyColorMap(sharpness_bgr, cv2.COLORMAP_MAGMA)
        
        # Upscale for visibility (480x640)
        radius_large = cv2.resize(radius_bgr, (580, 480), interpolation=cv2.INTER_NEAREST)
        sharpness_large = cv2.resize(sharpness_bgr, (640, 480), interpolation=cv2.INTER_NEAREST)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Overlay 15x20 Grid values on radius_large
        for r in range(15):
            for c in range(20):
                val = expected_radius[r, c]
                text = f"{val:.1f}"
                
                # Each cell is 580/20 = 29 wide, 480/15 = 32 high
                cx = int(c * 29.0 + 3)
                cy = int(r * 32.0 + 20)
                
                # Draw text with outline for visibility against heatmap colors
                cv2.putText(radius_large, text, (cx, cy), font, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(radius_large, text, (cx, cy), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
        # --- Create Vertical JET Colorbar (60x480) ---
        colorbar = np.zeros((480, 60, 3), dtype=np.uint8)
        # Create gradient from 0 to 255 (top is 255/value 9, bottom is 0/value 0)
        for y in range(480):
            val = int(255 - (y / 479) * 255)
            colorbar[y, :, :] = val
        colorbar = cv2.applyColorMap(colorbar, cv2.COLORMAP_JET)
        
        # Draw ticks 0..9 on Colorbar
        for tick in range(10):
            y = int(479 - (tick / 9.0) * 479)
            if y == 480: y = 479
            cv2.putText(colorbar, str(tick), (10, y + 5 if tick < 9 else y + 15), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(colorbar, str(tick), (10, y + 5 if tick < 9 else y + 15), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.line(colorbar, (0, y), (5, y), (255, 255, 255), 2)
            
        # Combine Radius Plot with its Colorbar (580 + 60 = 640 width)
        radius_large_with_bar = np.hstack((radius_large, colorbar))
        
        # Create a blank image for the 4th quadrant (bottom right)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # OpenCV text drawing
        t1 = f"Idx: {current_idx}/{total_images-1} | Scene: {scene_idx} | Camera Focus (Step): {step_idx}"
        t2 = f"Target GT Depth: {label} | [{key_str}]"
        
        # Draw labels on the plots
        cv2.putText(radius_large_with_bar, "Expected Radius (0-9)", (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(sharpness_large, f"Sharpness (Normalized by Max {max_sharpness:.2f})", (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(orig_bgr, "Original Composited Image", (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(orig_bgr, "[Press Q:Quit, B:Prev, Any:Next]", (10, 460), font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Combine the images into a 2x2 grid
        # Top row: Radius with Bar (Left), Sharpness (Right)
        top_row = np.hstack((radius_large_with_bar, sharpness_large))
        
        # Bottom row: Original Image (Centered, so we pad it with half blanks on sides for a hack, or just put it left and blank right)
        # To strictly center one image on the bottom of two images, it needs 320px black padding on each side.
        blank_half = np.zeros((480, 320, 3), dtype=np.uint8)
        bottom_row = np.hstack((blank_half, orig_bgr, blank_half))
        
        grid = np.vstack((top_row, bottom_row))
        
        # Title Overlay 
        overlay = grid.copy()
        cv2.rectangle(overlay, (0, 0), (1280, 80), (0, 0, 0), -1)
        grid = cv2.addWeighted(overlay, 0.6, grid, 0.4, 0)
        
        cv2.putText(grid, t1, (10, 30), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(grid, t2, (10, 60), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 4. Display and Handle Input
        # Make the window fit the screen nicely (1280x960 is quite big, resize for display if needed)
        display_grid = cv2.resize(grid, (960, 720), interpolation=cv2.INTER_AREA)
        cv2.imshow('Dataset Visualizer', display_grid)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            current_idx = max(0, current_idx - 1)
        else:
            current_idx = min(total_images - 1, current_idx + 1)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_dataset()
