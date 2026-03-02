import torch
import torch.nn.functional as F
import numpy as np
import cv2

from sfr_processor import get_processor

# --- Constants & Configuration ---
TARGET_H, TARGET_W = 480, 640
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GPU_PSF_GRIDS = {}

def init():
    print(f"Pre-calculating mock PSF grids on GPU (Device: {DEVICE})...")
    processor = get_processor(device=DEVICE)
    for r in range(0, 10):
        GPU_PSF_GRIDS[r] = processor.get_mock_psf_grid(radius=r, grid_h=3, grid_w=3)

def field_dependent_convolution(img_tensor, psf_grid):
    """
    Applies spatially-variant convolution using a grid of PSFs.
    img_tensor: [1, 1, H, W]
    psf_grid: 2D list of PyTorch tensors [grid_h][grid_w]
    """
    B, C, H, W = img_tensor.shape
    grid_h = len(psf_grid)
    grid_w = len(psf_grid[0])
    
    convolved_imgs = [[None for _ in range(grid_w)] for _ in range(grid_h)]
    
    for y in range(grid_h):
        for x in range(grid_w):
            k_tensor = psf_grid[y][x]
            k_h, k_w = k_tensor.shape[2:]
            pad_h = k_h // 2
            pad_w = k_w // 2
            padded = F.pad(img_tensor, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            out_tensor = F.conv2d(padded, k_tensor, padding=0)
            convolved_imgs[y][x] = out_tensor
            
    # Spatial Blending
    y_coords = torch.linspace(0, grid_h - 1, H, device=DEVICE).view(1, 1, H, 1)
    x_coords = torch.linspace(0, grid_w - 1, W, device=DEVICE).view(1, 1, 1, W)
    
    final_img = torch.zeros_like(img_tensor)
    weight_sum = torch.zeros_like(img_tensor)
    
    for y in range(grid_h):
        for x in range(grid_w):
            # Bilinear interpolation weights
            weight_y = torch.clamp(1.0 - torch.abs(y_coords - y), min=0.0)
            weight_x = torch.clamp(1.0 - torch.abs(x_coords - x), min=0.0)
            weight = weight_y * weight_x
            
            final_img += convolved_imgs[y][x] * weight
            weight_sum += weight
            
    final_img = final_img / (weight_sum + 1e-6)
    return final_img

def core_blur(clear_img, a, b):
    a = np.array(a)
    b = np.array(b)
    radii = (b - a).astype(int)
    
    img_tensor = torch.from_numpy(clear_img).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(img_tensor, size=(TARGET_H, TARGET_W), mode='bicubic', align_corners=False)
    
    outputs = []
    
    for r in radii:
        r = int(max(0, r))
        psf_grid = GPU_PSF_GRIDS.get(r)
        
        if psf_grid is not None:
            # Field-dependent blur is ALWAYS applied (even r=0 to capture baseline lens aberrations)
            out_tensor = field_dependent_convolution(resized, psf_grid)
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