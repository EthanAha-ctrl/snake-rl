import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def model_and_visualize(path="error_metrics.pkl"):
    if not os.path.exists(path):
        print(f"Error file not found at {path}. Run evaluate.py first to generate it.")
        return

    # 1. Load Data
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Convert to numpy arrays explicitly
    pos_errors = np.array(data.get("pos_errors", []))
    angle_errors_deg = np.array(data.get("angle_errors_deg", []))

    if len(pos_errors) == 0 or len(angle_errors_deg) == 0:
        print("No error data found in file.")
        return

    # 2. Model (Mean & Covariance)
    mean_vector = [np.mean(pos_errors), np.mean(angle_errors_deg)]
    cov_matrix = np.cov(pos_errors, angle_errors_deg)

    print(f"Original Mean: {mean_vector}")
    print(f"Original Covariance:\n{cov_matrix}")

    # 3. Generate Data
    # Generate same number of samples as original for fair comparison
    num_samples = len(pos_errors)
    generated_data = np.random.multivariate_normal(mean_vector, cov_matrix, num_samples)
    gen_pos_errors = generated_data[:, 0]
    gen_angle_errors_deg = generated_data[:, 1]

    # Calculate common ranges for visualization comparison
    all_pos = np.concatenate([pos_errors, gen_pos_errors])
    all_angle = np.concatenate([angle_errors_deg, gen_angle_errors_deg])
    
    pos_min, pos_max = np.min(all_pos), np.max(all_pos)
    angle_min, angle_max = np.min(all_angle), np.max(all_angle)
    
    # Add 5% padding
    pos_pad = (pos_max - pos_min) * 0.05
    angle_pad = (angle_max - angle_min) * 0.05
    
    pos_xlim = (pos_min - pos_pad, pos_max + pos_pad)
    angle_xlim = (angle_min - angle_pad, angle_max + angle_pad)

    # 4. Visualize
    plt.figure(figsize=(18, 10))

    # --- Row 1: Original Data ---
    
    # Col 1: Pos Hist
    plt.subplot(2, 3, 1)
    plt.hist(pos_errors, bins=50, color="blue", alpha=0.7)
    plt.title("Original: Position Error Dist")
    plt.xlabel("Error (position units)")
    plt.ylabel("Frequency")
    plt.xlim(pos_xlim)

    # Col 2: Angle Hist
    plt.subplot(2, 3, 2)
    plt.hist(angle_errors_deg, bins=50, color="green", alpha=0.7)
    plt.title("Original: Angle Error Dist")
    plt.xlabel("Error (degrees)")
    plt.ylabel("Frequency")
    plt.xlim(angle_xlim)

    # Col 3: Scatter
    plt.subplot(2, 3, 3)
    plt.scatter(pos_errors, angle_errors_deg, alpha=0.5, s=10, color="purple")
    cov_val_orig = cov_matrix[0, 1]
    plt.title(f"Original: Pos vs Angle\nCov: {cov_val_orig:.4f}")
    plt.xlabel("Position Error")
    plt.ylabel("Angle Error (deg)")
    plt.xlim(pos_xlim)
    plt.ylim(angle_xlim)
    plt.grid(True, alpha=0.3)

    # --- Row 2: Generated Data ---
    
    # Calculate stats for generated data
    gen_cov_matrix = np.cov(gen_pos_errors, gen_angle_errors_deg)
    gen_cov_val = gen_cov_matrix[0, 1]

    # Col 1: Pos Hist
    plt.subplot(2, 3, 4)
    plt.hist(gen_pos_errors, bins=50, color="blue", alpha=0.7)
    plt.title("Generated: Position Error Dist")
    plt.xlabel("Error (position units)")
    plt.ylabel("Frequency")
    plt.xlim(pos_xlim)

    # Col 2: Angle Hist
    plt.subplot(2, 3, 5)
    plt.hist(gen_angle_errors_deg, bins=50, color="green", alpha=0.7)
    plt.title("Generated: Angle Error Dist")
    plt.xlabel("Error (degrees)")
    plt.ylabel("Frequency")
    plt.xlim(angle_xlim)

    # Col 3: Scatter
    plt.subplot(2, 3, 6)
    plt.scatter(gen_pos_errors, gen_angle_errors_deg, alpha=0.5, s=10, color="purple")
    plt.title(f"Generated: Pos vs Angle\nCov: {gen_cov_val:.4f}")
    plt.xlabel("Position Error")
    plt.ylabel("Angle Error (deg)")
    plt.xlim(pos_xlim)
    plt.ylim(angle_xlim)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_and_visualize()
