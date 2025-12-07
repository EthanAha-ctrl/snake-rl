import pickle
import matplotlib.pyplot as plt
import os
import numpy as np


def visualize_error_file(path="error_metrics.pkl"):
    if not os.path.exists(path):
        print(f"Error file not found at {path}. Run evaluate.py first to generate it.")
        return

    with open(path, "rb") as f:
        data = pickle.load(f)

    pos_errors = data.get("pos_errors", [])
    angle_errors_deg = data.get("angle_errors_deg", [])

    if not pos_errors or not angle_errors_deg:
        print("No error data found in file.")
        return

    pos_mean = np.mean(pos_errors)
    angle_mean = np.mean(angle_errors_deg)
    pos_threshold = np.quantile(np.abs(pos_errors - pos_mean), 0.97)
    angle_threshold = np.quantile(np.abs(angle_errors_deg - angle_mean), 0.97)
    print(f"97% position error within: {pos_mean - pos_threshold:.4f} to {pos_mean + pos_threshold:.4f}")
    print(f"97% angle error (deg) within: {angle_mean - angle_threshold:.4f} to {angle_mean + angle_threshold:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(pos_errors, bins=50, color="blue", alpha=0.7)
    plt.title("Position Error Distribution")
    plt.xlabel("Error (position units)")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(angle_errors_deg, bins=50, color="green", alpha=0.7)
    plt.title("Angle Error Distribution")
    plt.xlabel("Error (degrees)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_error_file()
