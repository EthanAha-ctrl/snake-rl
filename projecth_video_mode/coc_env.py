import numpy as np
import my_gym
import json
import random

class CoCEnv(my_gym.Env):
    """
    CoC: Guess a Number Environment.
    Range: [0, 1]
    Objective: Guess the ground_truth number.
    Action: Float [0, 1]
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.min_val = 0.0
        self.max_val = 1.0
        
        # Action space: Continuous value in [0, 1]
        self.action_space = my_gym.Box(
            low=np.array([0.0, 0.0], dtype=np.float32), 
            high=np.array([1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # Observation space: Absolute difference in [0, 1]
        self.observation_space = my_gym.Box(
            low=np.array([0.0], dtype=np.float32), 
            high=np.array([1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        self.ground_truth = np.random.uniform(self.min_val, self.max_val)
        self.max_steps = 10
        self.current_step = 0
        self.target_step = np.random.uniform(self.min_val, self.max_val)
        self.diff_threshold = (self.max_val - self.min_val) / self.max_steps
        self.reached = False
        self.is_first_trial = True

        self.coc_error_dist = self._load_coc_error_dist("coc_error_distribution.json")

    def _load_coc_error_dist(self, path):
        with open(path, "r") as f:
            payload = json.load(f)
        dist = payload.get("error_distribution", None)
        if dist is None:
            raise ValueError("Invalid error distribution")
        
        out = {}
        for gt_str, pred_map in dist.items():
            gt = int(gt_str)
            out[gt] = {int(p): float(prob) for p , prob in pred_map.items()}
        return out

    def __calculate_noised_diff(self, diff:float) -> float:
        val = diff * 10
        idx_floor = int(np.floor(val))
        idx_ceil = idx_floor + 1
        
        weight_ceil = val - idx_floor
        weight_floor = 1.0 - weight_ceil
        
        idx_floor = int(np.clip(idx_floor, 0, 9))
        idx_ceil = int(np.clip(idx_ceil, 0, 9))

        dist_floor = self.coc_error_dist.get(idx_floor, {})
        dist_ceil = self.coc_error_dist.get(idx_ceil, {})
        
        merged_dist = {}
        for k, p in dist_floor.items():
            merged_dist[k] = merged_dist.get(k, 0.0) + float(p) * weight_floor
            
        for k, p in dist_ceil.items():
            merged_dist[k] = merged_dist.get(k, 0.0) + float(p) * weight_ceil
            
        keys = list(merged_dist.keys())
        weights = [merged_dist[k] for k in keys]
        sampled = random.choices(keys, weights=weights, k=1)[0]
        return sampled * 0.1

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.ground_truth = np.random.uniform(self.min_val, self.max_val)
        self.current_step = 0
        self.target_step = np.random.uniform(self.min_val, self.max_val)
        self.prev_diff = abs(self.target_step - self.ground_truth)
        self.reached = False
        self.is_first_trial = True
        return np.array([max(0.0, min(1.0, self.__calculate_noised_diff(self.prev_diff)))], dtype=np.float32), {}

    def step(self, command):
        if isinstance(command, np.ndarray):
            guess = float(command[0])
            action_val = float(command[1])
            action = (action_val > 0.5)
        else:
            guess, action = command
        
        if action:
            self.target_step = guess
        else:
            guess = self.target_step

        guess = max(0.0, min(1.0, guess))
        absolute_diff = abs(guess - self.ground_truth)
        
        self.current_step += 1
        
        r_guess = 0.0
        terminated = False
        
        improvement = self.prev_diff - absolute_diff
        r_guess = 1.0 if improvement >= 0 else -1
        r_guess -= (self.current_step * self.current_step) / 10.0
        r_trigger = 0.0

        reached = False
        if absolute_diff < self.diff_threshold:
            reached = True
            if self.reached == False:
                r_guess += 10.0
            else:
                r_guess = 0.0

        if self.current_step >= self.max_steps:
            terminated = True
            r_guess -= 10.0
            r_trigger -= 10.0

        if reached == False:
            if action == False:
                r_trigger -= (self.current_step * self.current_step)
            else:
                r_trigger += 1.0
        else:
            if self.reached == True:
                pass
            else:
                # just reached for the first time
                if action == False:
                    r_trigger -= (self.current_step * self.current_step)
                else:
                    r_trigger += 1.0

        if self.reached == True:
            if action == False:
                r_trigger += 1.0
                terminated = True
            else:
                r_trigger = -10.0

        if self.is_first_trial == True:
            self.is_first_trial = False
            r_guess = 0.1

        self.prev_diff = absolute_diff
        
        total_reward = np.array([r_guess, r_trigger], dtype=np.float32)
        
        if absolute_diff < self.diff_threshold:
            self.reached = True

        return np.array([max(0.0, min(1.0, self.__calculate_noised_diff(self.prev_diff)))], dtype=np.float32), total_reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            # 400x400 empty canvas
            return np.zeros((400, 400, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            pass
            
    def close(self):
        pass
