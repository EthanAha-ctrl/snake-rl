import os
import cv2
import numpy as np
import blur_ops

def create_random_polygon_mask(h, w):
    """
    Creates a random 20-vertex polygon mask.
    """
    # 1. Random radius (20% to 50% of image height)
    min_r = int(h * 0.2)
    max_r = int(h * 0.5)
    R = np.random.randint(min_r, max_r)
    
    # 2. Random center
    cx = np.random.randint(R, w - R)
    cy = np.random.randint(R, h - R)
    
    # 3. 20 vertices with perturbation
    num_vertices = 20
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    # Significant perturbation: -40% to +40% of R
    perturbations = np.random.uniform(-0.4, 0.4, size=num_vertices)
    r_perturbed = R * (1 + perturbations)
    
    pts_x = cx + r_perturbed * np.cos(angles)
    pts_y = cy + r_perturbed * np.sin(angles)
    pts = np.vstack((pts_x, pts_y)).T.astype(np.int32)
    
    # 4. Create binary mask (0 for bg, 255 for fg to be visually verifiable, but will use as 0/1 later)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    
    return mask

def get_random_fg_image(data_dir, current_bg_name, image_files, h, w):
    import random
    fg_candidates = [f for f in image_files if f != current_bg_name]
    if not fg_candidates:
        return None
    
    fg_name = random.choice(fg_candidates)
    fg_path = os.path.join(data_dir, fg_name)
    fg_bgr = cv2.imread(fg_path)
    if fg_bgr is None:
        return None
        
    fg_gray = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2GRAY)
    
    # Rotate if portrait and background is landscape, or vice versa
    fh, fw = fg_gray.shape
    if (fh > fw and h < w) or (fh < fw and h > w):
        fg_gray = cv2.rotate(fg_gray, cv2.ROTATE_90_CLOCKWISE)
        
    # Resize to match background dimensions exactly
    fg_resized = cv2.resize(fg_gray, (w, h), interpolation=cv2.INTER_AREA)
    return fg_resized, fg_name

def main():
    blur_ops.init()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data", "val2017") # Assuming validation set is here, adjust if needed
    
    if not os.path.exists(data_dir):
        # Fallback to pure data dir if val2017 doesn't exist
        data_dir = os.path.join(script_dir, "data")
        
    if not os.path.exists(data_dir):
        print(f"Error: Could not find data directory at {data_dir}")
        return
        
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {data_dir}")
        return
        
    # Sort for consistent ordering
    image_files.sort()
        
    print(f"Found {len(image_files)} images in {data_dir}.")
    print("Controls during display: 'n' (Next image), 'q' (Quit program), 'b' (Back focus), any other key (Next focus).")
    
    for img_name in image_files:
        img_path = os.path.join(data_dir, img_name)
        print(f"\nProcessing Background {img_name}...")
        
        bg_bgr = cv2.imread(img_path)
        if bg_bgr is None:
            print(f"Failed to read {img_path}, skipping.")
            continue
            
        bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
        
        # Rotate if portrait
        h, w = bg_gray.shape
        if h > w:
            bg_gray = cv2.rotate(bg_gray, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h
            
        # Get random foreground image
        fg_data = get_random_fg_image(data_dir, img_name, image_files, h, w)
        if fg_data is None:
            print("Failed to get foreground image, using background as fallback.")
            fg_gray = bg_gray.copy()
            fg_name = img_name
        else:
            fg_gray, fg_name = fg_data
            print(f"Loaded Foreground {fg_name}")
            
        print("Generating random mask...")
        mask = create_random_polygon_mask(h, w)
        
        # Focus levels / proxies
        a_list = np.arange(10)
        b_depth = 2 # Foreground focus plane proxy
        c_depth = 8 # Background focus plane proxy
        
        fg_radii = np.abs(a_list - b_depth).astype(int)
        bg_radii = np.abs(a_list - c_depth).astype(int)
        zeros = np.zeros_like(a_list)
        
        # Pre-multiply mask with foreground image to get foreground energy (0 outside mask)
        # Note: True optical blending must happen in linear light space!
        # Convert sRGB (approx) to linear light space by applying gamma 2.2
        fg_gray_linear = np.power(fg_gray.astype(np.float32) / 255.0, 2.2) * 255.0
        bg_gray_linear = np.power(bg_gray.astype(np.float32) / 255.0, 2.2) * 255.0
        
        fg_premult_img_linear = fg_gray_linear * (mask.astype(np.float32) / 255.0)
        
        print("Generating blur sequences (Optical Blending in Linear Space)...")
        # 1. Blur the foreground linear energy
        fg_blurred_imgs, _ = blur_ops.core_blur(fg_premult_img_linear, zeros, fg_radii)
        
        # 2. Blur the background linear image
        bg_imgs, _ = blur_ops.core_blur(bg_gray_linear, zeros, bg_radii)
        
        # 3. Blur the mask to get foreground alpha (occlusion strength)
        mask_blurred, _ = blur_ops.core_blur(mask, zeros, fg_radii)
        
        vis_list = []
        print("Blending foreground and background (Optical Accumulation)...")
        for i in range(10):
            # Result arrays from core_blur are uint8 0-255, which represent linear energy scaled to 0-255
            fg_energy_linear = fg_blurred_imgs[i].astype(np.float32) / 255.0
            bg_lum_linear = bg_imgs[i].astype(np.float32) / 255.0
            
            # Mask is 0-255, scale to 0-1 for alpha (occlusion)
            fg_alpha = mask_blurred[i].astype(np.float32) / 255.0
            
            # Physics-based blending in linear space: 
            blended_linear = fg_energy_linear + bg_lum_linear * (1.0 - fg_alpha)
            
            # Convert back to sRGB space by applying inverse gamma (1/2.2)
            blended_srgb = np.power(np.clip(blended_linear, 0, 1.0), 1.0 / 2.2) * 255.0
            blended = blended_srgb.astype(np.uint8)
            
            cv2.putText(blended, f"Focus (A): {a_list[i]}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(blended, f"FG (B): {b_depth} (radius={fg_radii[i]})", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(blended, f"BG (C): {c_depth} (radius={bg_radii[i]})", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(blended, f"BG: {img_name} | FG: {fg_name}", (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            
            vis_list.append(blended)
            
        idx = 0
        skip_to_next_image = False
        quit_program = False
        
        while True:
            idx = max(0, min(idx, len(vis_list) - 1))
            cv2.imshow("Foreground & Background Blur Test", vis_list[idx])
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                quit_program = True
                break
            elif key == ord('n'):
                skip_to_next_image = True
                break
            elif key == ord('b'):
                idx -= 1
            else:
                idx += 1
                if idx >= len(vis_list):
                   idx = len(vis_list) - 1
                   
        if quit_program:
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
