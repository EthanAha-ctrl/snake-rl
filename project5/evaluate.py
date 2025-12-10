import numpy as np
import torch
import my_gym

from history_stacker import HistoryStacker
from ppo_trainer import ActorCritic
from coc_env import CoCEnv

history_len = 4
use_random_action = False  # True = random actions, False = model actions
render_mode = "human"

env = CoCEnv(render_mode=render_mode)

obs, _ = env.reset()

# Obs dim for CoC is 1 (Box(1))
obs_dim = 1
if hasattr(env.observation_space, 'shape'):
    obs_dim = env.observation_space.shape[0]

stacker = HistoryStacker(obs_dim=obs_dim, history_len=history_len)
stacker.reset(obs)

stacked_dim = obs_dim * history_len + history_len

# Determine Action Dim and Continuous Flag
is_continuous = isinstance(env.action_space, my_gym.Box)
action_dim = 1 if is_continuous else env.action_space.n
if is_continuous and hasattr(env.action_space, 'shape'):
     action_dim = env.action_space.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(input_dim=stacked_dim, action_dim=action_dim, is_continuous=is_continuous).to(device)

try:
    state_dict = torch.load("ppo_coc.pth", map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded model from ppo_coc.pth")
except FileNotFoundError:
    print("Warning: ppo_coc.pth not found, using random weights")

model.eval()

for step in range(100):
    stacked_obs = torch.as_tensor(stacker.stacked(), dtype=torch.float32, device=device)
    
    if use_random_action:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            dist, _ = model(stacked_obs)
            if is_continuous:
                # For continuous, usually use mean for deterministic eval
                action_tensor = dist.mean
                action = action_tensor.item()
            else:
                action = dist.probs.argmax().item()
            
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step: {step}, Ground Truth: {env.ground_truth:.4f}, Action: {action:.4f}, Diff: {next_obs[0]:.4f}, Reward: {reward}")
    
    env.render()
    
    if terminated or truncated:
        print("--- Episode Done ---")
        next_obs, _ = env.reset()
        stacker.reset(next_obs)
    else:
        stacker.append(next_obs, action)

env.close()
