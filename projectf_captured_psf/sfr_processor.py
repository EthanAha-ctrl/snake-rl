import numpy as np
import cv2
import torch

class SFRProcessor:
    def __init__(self, device='cpu'):
        self.device = device
        
    def multiframe_average(self, image_paths):
        """
        Mock implementation: Reads multiple raw SFR charts and averages them to reduce noise.
        """
        if not image_paths:
            return None
        
        print(f"Averaging {len(image_paths)} frames for denoising...")
        # Since we don't have real data, we just return a mock blank image or the first one if we want.
        # In reality:
        # sum_img = np.zeros_like(cv2.imread(image_paths[0]), dtype=np.float32)
        # for p in image_paths: sum_img += cv2.imread(p).astype(np.float32)
        # return (sum_img / len(image_paths)).astype(np.uint8)
        
        # Mocking an empty 1000x1000 image for now
        return np.zeros((1000, 1000, 3), dtype=np.uint8)

    def extract_psf_grid(self, sfr_image, grid_h=3, grid_w=3):
        """
        Mock implementation: Extracts a grid of PSFs from a processed SFR chart.
        Returns a nested list/array of PSF kernels.
        """
        print(f"Extracting {grid_h}x{grid_w} PSF grid from SFR chart...")
        # Mocking by just calling the get_mock_psf_grid directly
        return self.get_mock_psf_grid(radius=3, grid_h=grid_h, grid_w=grid_w)

    def generate_mock_psf(self, radius, grid_y, grid_x, grid_h, grid_w):
        """
        Generates a physically plausible mock PSF for a given grid position.
        Simulates aberrations by elongating the kernel towards the edges.
        """
        # Base blur scale - even at radius 0, there is a small base PSF
        base_sigma = max(0.5, radius * 1.5)
        k_size = int(max(3, float(radius * 4 + 1)))
        if k_size % 2 == 0:
            k_size += 1
            
        # Optional: Make base kernel size larger if radius > 0
        if radius == 0:
            k_size = 5
            
        center_y, center_x = grid_h / 2.0 - 0.5, grid_w / 2.0 - 0.5
        
        # Compute vector from optical center to this grid position
        dy = grid_y - center_y
        dx = grid_x - center_x
        dist_from_center = np.sqrt(dx**2 + dy**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Astigmatism / Coma simulation: stretch kernel radially at edges
        stretch_factor = 1.0 + (dist_from_center / (max_dist + 1e-6)) * 0.8
        
        kernel = np.zeros((k_size, k_size), dtype=np.float32)
        cy, cx = k_size // 2, k_size // 2
        
        # Create a rotated 2D Gaussian
        theta = np.arctan2(dy, dx)
        
        sigma_u = base_sigma * stretch_factor # Radial direction
        sigma_v = base_sigma # Tangential direction
        
        # If radius == 0, just slight base blur, less stretch
        if radius == 0:
            sigma_u, sigma_v = 0.8, 0.8
            
        for y in range(k_size):
            for x in range(k_size):
                u = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
                v = -(x - cx) * np.sin(theta) + (y - cy) * np.cos(theta)
                
                # Equation of 2D Gaussian
                val = np.exp(-(u**2 / (2 * sigma_u**2 + 1e-6) + v**2 / (2 * sigma_v**2 + 1e-6)))
                kernel[y, x] = val
                
        # Normalize
        k_sum = np.sum(kernel)
        if k_sum > 1e-6:
            kernel /= k_sum
            
        # Convert to tensor: [1, 1, H, W]
        k_tensor = torch.from_numpy(kernel).to(self.device).unsqueeze(0).unsqueeze(0)
        return k_tensor

    def get_mock_psf_grid(self, radius, grid_h=3, grid_w=3):
        """
        Returns a mock grid of PSFs for a specific focus radius.
        Output: a 2D list of PyTorch tensors [grid_h][grid_w]
        """
        psf_grid = []
        for y in range(grid_h):
            row = []
            for x in range(grid_w):
                row.append(self.generate_mock_psf(radius, y, x, grid_h, grid_w))
            psf_grid.append(row)
            
        return psf_grid

# Singleton instance
processor = None

def get_processor(device):
    global processor
    if processor is None:
        processor = SFRProcessor(device=device)
    return processor
