import time
import numpy as np
import torch
from typing import List, Dict

from sac_trainer import SACConfig, SACTrainer
from history_stacker import HistoryStacker
from coc_sharpness_env import CoCEnv

def make_env():
    return CoCEnv(render_mode="human")

def load_model(config, input_dim, action_dim):
    trainer = SACTrainer(config, obs_dim=input_dim, action_dim=action_dim)
    trainer.load("sac_coc_best.pth")
    return trainer

def evaluate():
    config = SACConfig(buffer_size=10) # Minimal buffer for evaluation
    env = make_env()

    obs_dim = 1
    if hasattr(env.observation_space, 'shape'):
        obs_dim = env.observation_space.shape[0]
        
    # CoC specific: continuous action
    action_dim = 1
    if hasattr(env.action_space, 'shape'):
         action_dim = env.action_space.shape[0]

    history_len = 10 
    
    # Input dim for network logic matches train.py
    input_dim = obs_dim * history_len + action_dim * history_len

    trainer = load_model(config, input_dim, action_dim)
    
    stacker = HistoryStacker(obs_dim=obs_dim, action_dim=action_dim, history_len=history_len)
    
    obs, _ = env.reset()
    stacker.reset(obs, default_obs=-1.0, default_action=-1.0)

    import matplotlib.pyplot as plt
    import os
    
    os.makedirs("eval_plots", exist_ok=True)

    for i in range(10):  # Run 10 episodes
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        
        # We need ground truth access for printing debug info if available
        ground_truth = env.ground_truth if hasattr(env, 'ground_truth') else None
        print(f"\n--- Episode {i+1} Start. GT: {ground_truth} ---")

        # History for plotting
        episode_actions = []
        episode_obs = []

        while not (terminated or truncated):
            # Predict
            stacked_obs = stacker.stacked()
            action = trainer.select_action(stacked_obs, evaluate=True)
                
            # Step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            
            # Record for plotting
            # action is typically [guess, trigger_prob] or just [guess] depending on dim
            # Let's assume action[0] is the guess/position we care about for X-axis
            # next_obs is [sharpness]
            
            current_guess = float(action[0])
            current_sharpness = float(next_obs[0])
            
            episode_actions.append(current_guess)
            episode_obs.append(current_sharpness)
            
            print(f"Step: {step}, Action: {action}, Sharpness: {current_sharpness:.4f}, Reward: {reward}")

            if terminated or truncated:
                print("--- Episode Done ---")
                
                # Plotting
                plt.figure(figsize=(8, 5))
                plt.scatter(episode_actions, episode_obs, c='blue', label='Steps')
                plt.plot(episode_actions, episode_obs, linestyle='--', alpha=0.5)
                
                # Mark Start and End
                if episode_actions:
                    plt.scatter(episode_actions[0], episode_obs[0], c='green', s=100, label='Start')
                    plt.scatter(episode_actions[-1], episode_obs[-1], c='red', s=100, label='End')
                
                # Mark Ground Truth and Thresholds if available
                if ground_truth is not None:
                    plt.axvline(x=ground_truth, color='r', linestyle=':', label=f'GT: {ground_truth:.2f}')
                    
                    # Threshold lines
                    threshold = env.diff_threshold
                    plt.axvline(x=ground_truth - threshold, color='orange', linestyle='--', alpha=0.5, label='Threshold')
                    plt.axvline(x=ground_truth + threshold, color='orange', linestyle='--', alpha=0.5)

                plt.title(f"Episode {i+1}: Action (Guess) vs Sharpness (Obs)")
                plt.xlabel("Action (Guess Position)")
                plt.ylabel("Observation (Sharpness)")
                plt.legend()
                plt.grid(True)
                plt.xlim(0, 1) # Guess is always 0-1
                
                save_path = os.path.join("eval_plots", f"episode_{i+1:02d}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"Saved plot to {save_path}")

                next_obs, _ = env.reset()
                stacker.reset(next_obs, default_obs=-1.0, default_action=-1.0)
            else:
                stacker.append(next_obs, action)

            # time.sleep(0.05) # Speed up

        print(f"Episode {i+1} Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    evaluate()
