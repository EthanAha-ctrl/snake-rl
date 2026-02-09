import time
import numpy as np
import torch
from typing import List, Dict

from sac_trainer import SACConfig, SACTrainer
from history_stacker import HistoryStacker, position_angle_from_obs
from coc_env import CoCEnv

def make_env():
    return CoCEnv(render_mode="human")

def load_model(config, input_dim, action_dim):
    trainer = SACTrainer(config, obs_dim=input_dim, action_dim=action_dim)
    trainer.load("sac_coc.pth")
    return trainer

def evaluate():
    config = SACConfig() # defaults are fine for eval
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

    for i in range(10):  # Run 10 episodes
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        
        # We need ground truth access for printing debug info if available
        ground_truth = env.ground_truth if hasattr(env, 'ground_truth') else None
        print(f"\n--- Episode {i+1} Start. GT: {ground_truth} ---")

        while not (terminated or truncated):
            env.render()
            
            # Predict
            stacked_obs = stacker.stacked()
            action = trainer.select_action(stacked_obs, evaluate=True)
            
            # Step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            
            print(f"Step: {step}, Action: {action}, Diff: {next_obs[0]}, Reward: {reward}")

            if terminated or truncated:
                print("--- Episode Done ---")
                next_obs, _ = env.reset()
                stacker.reset(next_obs, default_obs=-1.0, default_action=-1.0)
            else:
                stacker.append(next_obs, action)

            time.sleep(0.05)

        print(f"Episode {i+1} Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    evaluate()
