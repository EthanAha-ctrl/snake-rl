import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.fftpack import dct, idct
import os

def radial_base(x, y, cx, cy, radius_sigma):
    """
    Stage 1: Base radial Gaussian or disk model.
    Using a Gaussian approximation for the smooth base.
    """
    r_sq = (x - cx)**2 + (y - cy)**2
    # simple un-normalized gaussian for base energy
    return np.exp(-r_sq / (2 * radius_sigma**2))

def cosine_term(r, clamp, amp, freq, phase):
    """
    A single high-frequency cosine term with a hard clamp.
    r: radial distance from center
    """
    # Using np.where to handle the clamp in a vectorized way
    return np.where(r > clamp, 0.0, amp * np.cos(r * freq + phase))

def approximated_kernel(x, y, params, num_cosine_terms=11):
    """
    The full closed-form approximation: Radial Base + N * Cosine Terms.
    params layout:
    [0]: cx
    [1]: cy
    [2]: radius_sigma
    [3 + i*4]: amp_i
    [4 + i*4]: phase_i
    [5 + i*4]: freq_i
    [6 + i*4]: clamp_i
    """
    cx, cy, radius_sigma = params[0], params[1], params[2]
    
    # 1. Base Radial
    base_val = radial_base(x, y, cx, cy, radius_sigma)
    
    # 2. Add Cosine Terms
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    cosine_sum = np.zeros_like(r)
    
    offset = 3
    for i in range(num_cosine_terms):
        amp = params[offset]
        phase = params[offset + 1]
        freq = params[offset + 2]
        clamp = params[offset + 3]
        
        cosine_sum += cosine_term(r, clamp, amp, freq, phase)
        offset += 4
        
    return base_val + cosine_sum

def residual_error_func(params, x, y, target_img, num_cosine_terms):
    """
    The objective function for scipy.optimize.least_squares
    """
    pred = approximated_kernel(x, y, params, num_cosine_terms)
    # Return 1D array of residuals
    return (pred - target_img).ravel()

def extract_psf_from_led(image_path, crop_margin=1.2):
    """
    Pipeline 2B: Extract Point Spread Function from LED targets using OpenCV.
    Finds bright spots, crops with margin, and aligns center.
    """
    print(f"[OpenCV] Loading LED image: {image_path}")
    # Dummy implementation for structure
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # circles = cv2.HoughCircles(...) 
    # crop and align...
    
    # Returning a dummy synthetic 51x51 PSF for testing the optimizer
    grid_size = 51
    Y, X = np.mgrid[0:grid_size, 0:grid_size]
    # synthetic "target" with base + 1 ripple
    target = radial_base(X, Y, 25, 25, 12.0) + cosine_term(np.sqrt((X-25)**2 + (Y-25)**2), 15.0, 0.2, 0.8, 0.0)
    return target, X, Y
    
def generate_initial_guess_via_dct(residual_img, base_radius, num_cosine_terms):
    """
    Uses DCT on the windowed residual to find dominant frequencies
    for the optimizer's initial guess.
    """
    print(f"[DCT] Running DCT to find {num_cosine_terms} dominant frequencies...")
    # 1. Apply soft window using base_radius to suppress background noise
    # (Dummy logic for structural representation)
    h, w = residual_img.shape
    dct_result = dct(dct(residual_img.T, norm='ortho').T, norm='ortho')
    
    # 2. Extract top peaks (simplified to random/heuristic guesses for now)
    # Real implementation would do peak finding on dct_result
    
    guess_params = []
    for i in range(num_cosine_terms):
        amp = 0.05 / (i + 1) # Diminishing amplitude
        phase = 0.0
        # freq derived from DCT peak coordinates (placeholder)
        freq = 0.5 + i * 0.2 
        clamp = base_radius # Default clamp to edge of base
        guess_params.extend([amp, phase, freq, clamp])
        
    return guess_params

def fit_closed_form_psf(target_psf, X, Y, num_cosine_terms=11):
    """
    Two-Stage Optimizer for the Extracted Off-Focus PSF.
    """
    print("--- Stage 1: Base Radial Fitting ---")
    # Guess: center in middle, some radius
    cx_guess = X.max() / 2.0
    cy_guess = Y.max() / 2.0
    rad_guess = 10.0
    
    # [cx, cy, radius_sigma]
    stage1_guess = [cx_guess, cy_guess, rad_guess]
    
    # We pad the guess with zeros for the cosine terms during stage 1 to just fit the base
    # In reality, you'd fit a simpler model first.
    
    # Let's assume stage 1 found these optimal base params:
    base_params = [25.0, 25.0, 11.5] 
    
    print(f"Base fitted params: cx={base_params[0]:.2f}, cy={base_params[1]:.2f}, rad={base_params[2]:.2f}")
    
    print("--- Stage 2: Residual Multi-Term Cosine Fitting ---")
    base_pred = radial_base(X, Y, base_params[0], base_params[1], base_params[2])
    residual = target_psf - base_pred
    
    cosine_guess = generate_initial_guess_via_dct(residual, base_radius=base_params[2], num_cosine_terms=num_cosine_terms)
    
    # Full initial guess x0
    x0 = base_params + cosine_guess
    
    # Bounds setup: 
    # cx, cy, rad_sigma: (0, inf)
    # amp: (-inf, inf)
    # phase: (-pi, pi)
    # freq: (0, inf)
    # clamp: (0, image_size)
    
    # For simplicity in this scaffolding, we use unbounded or simple bounds
    print(f"[Optimizer] Launching Levenberg-Marquardt with {len(x0)} parameters...")
    
    # Run optimization (This is the heavy lifting)
    result = least_squares(
        residual_error_func, 
        x0, 
        args=(X, Y, target_psf, num_cosine_terms),
        method='lm', # Levenberg-Marquardt is great for unconstrained non-linear least squares
        max_nfev=500
    )
    
    print(f"[Optimizer] Finished. Success: {result.success}, Cost: {result.cost:.4f}")
    return result.x

def main():
    print("=== Imtest / LED PSF Parsing & Fitting Pipeline ===")
    
    # Simulating data extraction from LED crop
    target_psf, X, Y = extract_psf_from_led("dummy_led_image.png")
    
    # Fit the mathematical closed-form to our extracted target
    fitted_params = fit_closed_form_psf(target_psf, X, Y, num_cosine_terms=5) # using 5 for demo speed
    
    print("\n[Result] Final Extracted Param Vector layout (first 10):")
    print(np.round(fitted_params[:10], 3))
    print("... Ready to export to Tensor DB or use in RL forward pass!")

if __name__ == "__main__":
    main()
