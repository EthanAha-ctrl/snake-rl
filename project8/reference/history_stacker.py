from collections import deque
from typing import Deque
import numpy as np


# Only use position and angle from the raw CartPole observation
def position_angle_from_obs(obs: np.ndarray) -> np.ndarray:
    return np.array([obs[0], obs[2]], dtype=np.float32)


class HistoryStacker:
    def __init__(self, obs_dim: int, history_len: int):
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.obs_history: Deque[np.ndarray] = deque(maxlen=history_len)
        self.action_history: Deque[float] = deque(maxlen=history_len)

    def reset(self, initial_obs: np.ndarray, default_obs: float = 0.0, default_action: float = 0.0):
        self.obs_history.clear()
        self.action_history.clear()
        
        # Use default_obs value but respect obs_dim shape
        zero_obs = np.full(self.obs_dim, default_obs, dtype=np.float32)
        
        for _ in range(self.history_len - 1):
            self.obs_history.append(zero_obs)
            self.action_history.append(float(default_action))
        self.obs_history.append(initial_obs.astype(np.float32))
        self.action_history.append(float(default_action))

    def append(self, obs: np.ndarray, action: float):
        self.obs_history.append(obs.astype(np.float32))
        self.action_history.append(float(action))

    def stacked(self) -> np.ndarray:
        obs_stack = np.concatenate(list(self.obs_history), dtype=np.float32)
        act_stack = np.array(self.action_history, dtype=np.float32)
        return np.concatenate([obs_stack, act_stack], dtype=np.float32)
