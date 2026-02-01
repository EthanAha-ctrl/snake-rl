import gymnasium as gym
import numpy as np
import os
import tqdm
import json
import matplotlib.pyplot as plt

def generate_data(num_episodes=1000, data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.unwrapped.theta_threshold_radians = np.deg2rad(45.0)
    env.unwrapped.x_threshold = 1.5

    obs, _ = env.reset()
    initial_frame = env.render()
    img_shape = initial_frame.shape

    x_threshold = env.unwrapped.x_threshold

    metadata = {
        "img_shape": list(img_shape),
        "x_threshold": x_threshold,
    }

    with open(os.path.join(data_dir, "dataset_config.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    mmap_path = os.path.join(data_dir, "images.bin")
    if os.path.exists(mmap_path):
        os.remove(mmap_path)

    labels = []
    total_frames = 0

    with open(mmap_path, "wb") as f_out:
        for _ in tqdm.tqdm(range(num_episodes)):
            env.reset()

            random_state = np.random.uniform(low=[-1.5, -0.5, -0.8, -0.5],
                                           high=[1.5, 0.5, 0.8, 0.5])

            env.unwrapped.state = random_state

            done = False
            steps = 0
            while not done and steps < 100:
                image = env.render()
                f_out.write(image.tobytes())

                current_state = env.unwrapped.state
                labels.append([current_state[0], current_state[2]]) # Cart Pos, Pole Angle
                total_frames += 1

                action = env.action_space.sample()
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1

    os.rename(mmap_path, os.path.join(data_dir, "images.mmap"))

    np.save(os.path.join(data_dir, "labels.npy"), np.array(labels))

    print(f"Generated {total_frames} frames using Domain Randomization.")

def visualize_data(data_dir="data"):
    print("Loading data for visualization...")
    labels_path = os.path.join(data_dir, "labels.npy")
    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} not found.")
        return

    labels_arr = np.load(labels_path)

    print("Visualizing data distribution...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(labels_arr[:, 0], bins=50, color='blue', alpha=0.7)
    plt.title("Cart Position Distribution")
    plt.xlabel("Position")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(labels_arr[:, 1], bins=50, color='green', alpha=0.7)
    plt.title("Pole Angle Distribution")
    plt.xlabel("Angle (Rad)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #generate_data()
    visualize_data()