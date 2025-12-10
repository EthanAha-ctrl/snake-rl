import numpy as np
import torch
import gymnasium as gym
import pickle
import os

from history_stacker import position_angle_from_obs, HistoryStacker
from ppo_trainer import ActorCritic

history_len = 4
use_random_action = False  # True = random actions, False = model actions
render_mode = "human"


class NoiseInjector:
    def __init__(self, path="error_metrics.pkl"):
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Noise injection disabled.")
            self.active = False
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        pos_errors = np.array(data.get("pos_errors", []))
        angle_errors_deg = np.array(data.get("angle_errors_deg", []))

        if len(pos_errors) == 0:
            self.active = False
            return

        # Model as Multivariate Normal Distribution
        self.mean = np.array([np.mean(pos_errors), np.mean(angle_errors_deg)])
        self.cov = np.cov(pos_errors, angle_errors_deg)
        self.active = True
        print(f"Noise Injector initialized using {path}")
        print(f"Noise Mean: {self.mean}")
        print(f"Noise Covariance:\n{self.cov}")

    def add_noise(self, obs: np.ndarray) -> np.ndarray:
        # obs is [pos, angle_rad]
        if not self.active:
            return obs

        noise = np.random.multivariate_normal(self.mean, self.cov)
        # noise is [pos_err, angle_err_deg]

        noisy_obs = obs.copy()
        noisy_obs[0] += noise[0]
        noisy_obs[1] += np.deg2rad(noise[1])

        return noisy_obs


env = gym.make("CartPole-v1", render_mode=render_mode)
env.unwrapped.theta_threshold_radians = np.deg2rad(45.0)
env.unwrapped.x_threshold = 1.5

noise_injector = NoiseInjector("error_metrics.pkl")

obs, _ = env.reset()
obs = position_angle_from_obs(obs)
obs = noise_injector.add_noise(obs)

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
        next_obs = noise_injector.add_noise(next_obs)
        stacker.reset(next_obs)
    else:
        next_obs = position_angle_from_obs(next_obs)
        next_obs = noise_injector.add_noise(next_obs)
        stacker.append(next_obs, action)

env.close()
