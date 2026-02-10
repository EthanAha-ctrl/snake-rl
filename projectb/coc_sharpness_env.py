import numpy as np
import my_gym
import json
import random
import lmdb
import pickle
import torch
import os

# Tensor DB paths
META_PATH = os.path.join("reference", "project7", "data", "coc_meta.pkl")
LMDB_PATH = os.path.join("reference", "project7", "data", "coc_tensor_10x15x20.lmdb")

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
        
        # Observation space:
        self.observation_space = my_gym.Box(
            low=np.array([0.0], dtype=np.float32), 
            high=np.array([float('inf')], dtype=np.float32), 
            dtype=np.float32
        )
        
        self.ground_truth = np.random.uniform(self.min_val, self.max_val)
        self.max_steps = 10
        self.current_step = 0
        self.target_step = np.random.uniform(self.min_val, self.max_val)
        self.diff_threshold = (self.max_val - self.min_val) / self.max_steps
        self.reached = False
        self.is_first_trial = True

        self.meta_path = META_PATH
        self.lmdb_path = LMDB_PATH
        
        self._init_data()

    def _init_data(self):
        # Load Metadata
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata not found: {self.meta_path}")
            
        with open(self.meta_path, 'rb') as f:
            meta_info = pickle.load(f)
            
        # Build label to key map
        self.label_to_sharpness = {i: [] for i in range(10)}
        
        count = 0
        for item in meta_info:
            if len(item) == 3:
                _, label, sharpness_grid = item
                self.label_to_sharpness[int(label)].append(sharpness_grid)
                count += 1
            else:
                # Fallback for old format if any (shouldn't be based on project7)
                pass
        
        self.total_entries = count
        print(f"Loaded sharpness data for {self.total_entries} images.")

    def _get_random_sharpness_for_label(self, label):
        grids = self.label_to_sharpness[int(label)]
        if not grids:
            return np.zeros((15, 20), dtype=np.float32)
            
        # Use fixed index if available, else random (should be set in reset)
        assert hasattr(self, 'current_img_index'), "current_img_index not set"
        idx = self.current_img_index % len(grids)
        return grids[idx]

    def _get_interpolated_sharpness(self, val):
        val = np.clip(val, 0.0, 9.999)
        label_floor = int(np.floor(val))
        label_ceil = label_floor + 1
        
        weight_ceil = val - label_floor
        weight_floor = 1.0 - weight_ceil
        
        # Clamp labels
        label_floor = min(max(label_floor, 0), 9)
        label_ceil = min(max(label_ceil, 0), 9)
        
        t_floor = self._get_random_sharpness_for_label(label_floor)
        t_ceil = self._get_random_sharpness_for_label(label_ceil)
        
        # Weighted Average
        t_mix = t_floor * weight_floor + t_ceil * weight_ceil
        scalar_sharpness = np.mean(t_mix) / 640.0 / 480.0
        return np.array([scalar_sharpness], dtype=np.float32) * self.sharpness_scale

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_img_index = random.randint(0, self.total_entries)
        self.sharpness_scale = np.random.uniform(0.5, 1.5)
        self.ground_truth = np.random.uniform(self.min_val, self.max_val)
        self.current_step = 0
        self.target_step = np.random.uniform(self.min_val, self.max_val)
        self.prev_diff = abs(self.target_step - self.ground_truth)
        self.reached = False
        self.is_first_trial = True
        return self._get_interpolated_sharpness(self.prev_diff * 10.0), {}

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
        obs = self._get_interpolated_sharpness(absolute_diff * 10.0)
        
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
            if (reached == True) and (action == True):
                r_guess = 10.0

        self.prev_diff = absolute_diff
        
        total_reward = np.array([r_guess, r_trigger], dtype=np.float32)
        
        if absolute_diff < self.diff_threshold:
            self.reached = True

        return obs, total_reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            # 400x400 empty canvas
            return np.zeros((400, 400, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            pass
            
    def close(self):
        pass
