import numpy as np
import torch
import gymnasium as gym

from history_stacker import position_angle_from_obs, HistoryStacker
from ppo_trainer import ActorCritic

history_len = 4
use_random_action = False  # True = random actions, False = model actions
render_mode = "human"

env = gym.make("CartPole-v1", render_mode=render_mode)
env.unwrapped.theta_threshold_radians = np.deg2rad(45.0)

obs, _ = env.reset()
obs = position_angle_from_obs(obs)

stacker = HistoryStacker(obs_dim=obs.shape[0], history_len=history_len)
stacker.reset(obs)

stacked_dim = obs.shape[0] * history_len + history_len
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(input_dim=stacked_dim, action_dim=env.action_space.n).to(device)
state_dict = torch.load("ppo_cartpole.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

for _ in range(1000):
    stacked_obs = torch.as_tensor(stacker.stacked(), dtype=torch.float32, device=device)
    if use_random_action:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            logits, _ = model(stacked_obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.probs.argmax().item()
    next_obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        next_obs, _ = env.reset()
        next_obs = position_angle_from_obs(next_obs)
        stacker.reset(next_obs)
    else:
        stacker.append(position_angle_from_obs(next_obs), action)

env.close()
