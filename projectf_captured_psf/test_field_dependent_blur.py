import numpy as np
import cv2
import os
from blur_ops import core_blur, init

def main():
    print("Testing field-dependent blur simulation...")
    init()
    
    # Create a synthetic image (grid pattern)
    img = np.zeros((480, 640), dtype=np.uint8)
    img[::40, :] = 255
    img[:, ::40] = 255
    
    # Run core_blur
    a_list = [0] * 3
    b_list = list(range(0, 3))
    print(f"Running core_blur with a={a_list}, b={b_list}")
    
    blurred_batch, sharpness_batch = core_blur(img, a_list, b_list)
    
    print(f"Generated {len(blurred_batch)} blurred images.")
    
    # Save results
    os.makedirs("test_outputs", exist_ok=True)
    for i, b_img in enumerate(blurred_batch):
        filename = f"test_outputs/blur_radius_{b_list[i]}.png"
        cv2.imwrite(filename, b_img)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()
