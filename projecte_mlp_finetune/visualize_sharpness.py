import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Paths
META_PATH = os.path.join("data", "coc_meta.pkl")
OUTPUT_DIR = "sharpness_plots"

def main():
    if not os.path.exists(META_PATH):
        print(f"Error: Metadata file not found at {META_PATH}")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading metadata from {META_PATH}...")

    with open(META_PATH, 'rb') as f:
        meta_info = pickle.load(f)

    print(f"Total metadata entries: {len(meta_info)}")

    # In preprocessing.py, for each image, it loops i in range(10) (0 to 9) and appends to meta_info.
    # So every 10 entries belong to the same original image, in order of blur kernels.
    
    # Calculate number of unique source images
    num_images_to_plot = len(meta_info) // 10
    print(f"Plotting all {num_images_to_plot} unique images found in metadata...")
    
    # Process each image group
    for img_idx in range(num_images_to_plot):
        start_idx = img_idx * 10
        end_idx = start_idx + 10
        
        if end_idx > len(meta_info):
            break
            
        group = meta_info[start_idx:end_idx]
        
        # Verify labels are 0-9
        labels = []
        sharpness_values = []
        
        valid_group = True
        for i, item in enumerate(group):
            # item is likely (key_str, label, sharpness_grid)
            if len(item) == 3:
                key, label, grid = item
            else:
                print(f"Skipping group starting at {start_idx} due to unexpected format")
                valid_group = False
                break
                
            # Verify label matches index (0-9 sequence)
            # Actually preprocessing.py appends in order of b_list which is 0..9
            # But let's just trust 'label'
            
            # Calculate scalar sharpness
            # Formula from coc_sharpness_env.py: mean(grid) / 640.0 / 480.0
            scalar_s = np.mean(grid) / (640.0 * 480.0)
            
            labels.append(label)
            sharpness_values.append(scalar_s)
        
        if not valid_group:
            continue
            
        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(labels, sharpness_values, marker='o', linestyle='-')
        plt.title(f"Sharpness Decay for Image {img_idx}")
        plt.xlabel("Blur Label (0=Clear, 9=Blurred)")
        plt.ylabel("Scalar Sharpness (Normalized)")
        plt.grid(True)
        plt.xticks(range(10))
        
        # Save
        save_path = os.path.join(OUTPUT_DIR, f"sharpness_img_{img_idx:03d}.png")
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved plot to {save_path}")

    print("Done. Please check the output directory.")

if __name__ == "__main__":
    main()
