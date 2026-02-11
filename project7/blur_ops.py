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
        return np.array([]), []
        
    batch_out = torch.cat(outputs, dim=0)
    batch_out = batch_out.clamp(0, 255).byte()
    final_imgs = batch_out.squeeze(1).cpu().numpy()

    sharpness_list = []
    for img in final_imgs:
        s = calculate_sharpness_grid(img, kernel_size=5)
        sharpness_list.append(s)

    return final_imgs, sharpness_list


class BlurSimulatorStub:
    def __init__(self):
        pass

    def get_blur_range(self, anchor_idx):
        """
        Returns the list of blur levels needed for a given anchor.
        For CoC simulation, this is simply 0 to anchor_idx.
        """
        return list(range(anchor_idx + 1))

    def simulate_scene(self, clear_img, anchor_idx, mode='coc'):
        """
        Generates the blurred images for a specific anchor configuration.
        
        Args:
            clear_img: Input image (grayscale)
            anchor_idx: The anchor index (0-9) representing the lens configuration/max blur.
            mode: 'coc' for internal simulation, 'zemax' for external.
            
        Returns:
            List of (blurred_image, blur_level, sharpness_grid) tuples.
        """
        if mode == 'coc':
            # Scheme A: Explicit generation for this anchor
            # We want to generate images for blur levels 0 to anchor_idx
            
            target_levels = self.get_blur_range(anchor_idx)
            
            # For CoC, we always start from the clear image (blur=0)
            # So 'a' (start) is always 0. 'b' (end) is the target level.
            a_list = [0] * len(target_levels)
            b_list = target_levels
            
            blurred_imgs, sharpness_grids = core_blur(clear_img, a_list, b_list)
            
            results = []
            for i, level in enumerate(target_levels):
                results.append((blurred_imgs[i], level, sharpness_grids[i]))
            
            return results
            
        elif mode == 'zemax':
            print(f"Zemax mode for anchor {anchor_idx} not implemented. Returning empty.")
            return []
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

simulator = BlurSimulatorStub()


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