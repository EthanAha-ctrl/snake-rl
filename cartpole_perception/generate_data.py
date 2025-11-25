import gymnasium as gym
import numpy as np
import os
import tqdm
from stable_baselines3 import A2C

def generate_data(num_episodes=200, data_dir="data"):
    """
    Generates data for training the perception model using a two-pass approach
    with a memory-mapped file for efficient storage.

    Args:
        num_episodes (int): The number of episodes to run to generate data.
        data_dir (str): The directory to save the data in.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    # First, train an agent to solve the environment
    print("Training a control agent to generate good data...")
    model = A2C("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10_000)
    print("Control agent trained.")

    # --- Pass 1: Count total frames ---
    print("Pass 1: Counting total number of frames...")
    total_frames = 0
    for i in tqdm.tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_frames += 1
    
    print(f"Counted {total_frames} total frames to be generated.")

    # --- Pass 2: Generate data and save to memmap ---
    print("Pass 2: Generating data and writing to memory-mapped file...")
    
    # Get image shape from one render and save it
    img_shape = env.render().shape
    with open(os.path.join(data_dir, "metadata.txt"), "w") as f:
        f.write(",".join(map(str, img_shape)))
        
    # Create the memory-mapped file
    mmap_path = os.path.join(data_dir, "images.mmap")
    if os.path.exists(mmap_path):
        os.remove(mmap_path) # Ensure we start fresh
    images_mmap = np.memmap(mmap_path, dtype=np.uint8, mode='w+', shape=(total_frames, *img_shape))
    
    labels = []
    frame_idx = 0
    for i in tqdm.tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            image = env.render()
            
            cart_position = state[0]
            pole_angle = state[2]

            # Write directly to the memory-mapped array
            images_mmap[frame_idx] = image
            labels.append([cart_position, pole_angle])
            
            frame_idx += 1
            obs = state
            
    # Flush memory map to disk and save labels
    images_mmap.flush()
    np.save(os.path.join(data_dir, "labels.npy"), np.array(labels))

    print(f"Generated {frame_idx} images and labels.")
    print(f"Image data saved to {mmap_path}")
    print(f"Label data saved to {os.path.join(data_dir, 'labels.npy')}")


if __name__ == "__main__":
    generate_data()