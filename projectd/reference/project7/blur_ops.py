import torch
import torch.nn.functional as F
import numpy as np
import cv2

# --- Constants & Configuration ---
TARGET_H, TARGET_W = 480, 640
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GPU_KERNELS = {}

def generate_kernel_tensor(radius):
    if radius == 0:
        return torch.ones((1, 1, 1, 1), device=DEVICE)

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
        
    k_tensor = torch.from_numpy(kernel).to(DEVICE).unsqueeze(0).unsqueeze(0)
    return k_tensor

def init():
    print(f"Pre-calculating kernels on GPU (Device: {DEVICE})...")
    for r in range(0, 10):
        GPU_KERNELS[r] = generate_kernel_tensor(r)

def core_blur(clear_img, a, b):
    a = np.array(a)
    b = np.array(b)
    radii = (b - a).astype(int)
    
    img_tensor = torch.from_numpy(clear_img).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(img_tensor, size=(TARGET_H, TARGET_W), mode='bicubic', align_corners=False)
    
    outputs = []
    
    for r in radii:
        k_tensor = GPU_KERNELS.get(r)
        
        if r > 0 and k_tensor is not None:
            k_h, k_w = k_tensor.shape[2:]
            pad_h = k_h // 2
            pad_w = k_w // 2
            padded = F.pad(resized, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            out_tensor = F.conv2d(padded, k_tensor, padding=0)
        else:
            out_tensor = resized
        
        outputs.append(out_tensor)
        
    if not outputs:
        return np.array([])
        
    batch_out = torch.cat(outputs, dim=0)
    batch_out = batch_out.clamp(0, 255).byte()
    final_imgs = batch_out.squeeze(1).cpu().numpy()

    sharpness_list = []
    for img in final_imgs:
        s = calculate_sharpness_grid(img, kernel_size=5)
        sharpness_list.append(s)

    return final_imgs, sharpness_list


def calculate_sharpness_grid(img, kernel_size=5):
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=kernel_size)
    abs_lap = np.abs(lap)
    
    h, w = img.shape
    block_h = h / 15.0
    block_w = w / 20.0
    block_area = block_h * block_w
    
    means = cv2.resize(abs_lap, (20, 15), interpolation=cv2.INTER_AREA)
    sums = means * block_area
    return sums.astype(np.float32)

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
    
    # 4. Create binary mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    
    return mask

def generate_focus_stack(bg_gray, fg_gray, a_list, b_depth, c_depth):
    """
    Generates a list of 10 optically blended images based on focus distance 'a'.
    
    Args:
        bg_gray: Background image (grayscale, uint8).
        fg_gray: Foreground image (grayscale, uint8, same size as bg_gray).
        a_list: List of 10 focus proxies [0, ..., 9].
        b_depth: Foreground focus proxy.
        c_depth: Background focus proxy.
        
    Returns:
        blended_imgs: List of 10 blended grayscale numpy images (TARGET_H x TARGET_W, uint8).
        labels: List of 10 absolute foreground blur radii for each image.
    """
    h, w = bg_gray.shape
    mask = create_random_polygon_mask(h, w)
    
    fg_radii = np.abs(np.array(a_list) - b_depth).astype(int)
    bg_radii = np.abs(np.array(a_list) - c_depth).astype(int)
    zeros = np.zeros_like(a_list)
    
    # Convert sRGB (approx) to linear light space by applying gamma 2.2
    fg_gray_linear = np.power(fg_gray.astype(np.float32) / 255.0, 2.2) * 255.0
    bg_gray_linear = np.power(bg_gray.astype(np.float32) / 255.0, 2.2) * 255.0
    
    fg_premult_img_linear = fg_gray_linear * (mask.astype(np.float32) / 255.0)
    
    # Blur the foreground linear energy
    fg_blurred_imgs, _ = core_blur(fg_premult_img_linear, zeros, fg_radii)
    
    # Blur the background linear image
    bg_imgs, _ = core_blur(bg_gray_linear, zeros, bg_radii)
    
    # Blur the mask to get foreground alpha (occlusion strength)
    mask_blurred, _ = core_blur(mask, zeros, fg_radii)
    
    blended_imgs = []
    labels = []
    
    for i in range(10):
        # Result arrays from core_blur are uint8 0-255, scale back to 0-1 for precision blending
        fg_energy_linear = fg_blurred_imgs[i].astype(np.float32) / 255.0
        bg_lum_linear = bg_imgs[i].astype(np.float32) / 255.0
        
        # Mask is 0-255, scale to 0-1 for alpha (occlusion)
        fg_alpha = mask_blurred[i].astype(np.float32) / 255.0
        
        # Physics-based blending in linear space
        blended_linear = fg_energy_linear + bg_lum_linear * (1.0 - fg_alpha)
        
        # Convert back to sRGB space by applying inverse gamma (1/2.2)
        blended_srgb = np.power(np.clip(blended_linear, 0, 1.0), 1.0 / 2.2) * 255.0
        blended = blended_srgb.astype(np.uint8)
        
        blended_imgs.append(blended)
        labels.append(int(fg_radii[i])) # The label is the absolute blur radius of the foreground
        
    return blended_imgs, labels