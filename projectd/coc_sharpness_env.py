import numpy as np
import my_gym
import json
import random
import lmdb
import pickle
import torch
import os

# Tensor DB paths
META_PATH = os.path.join("reference", "project7", "data", "coc_meta_foreground_background.pkl")
LMDB_PATH = os.path.join("reference", "project7", "data", "coc_tensor_10x15x20.lmdb")

class CoCEnv(my_gym.Env):
    """
    CoC: Guess a Number Environment.
    Range: [0, 1]
    Objective: Guess the ground_truth number.
    Action: Float [0, 1]
    
    Observation: [Sharpness, Expected Radius]
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
        
        # Observation space: 10x15x20 flattened vision + 15x20 flattened sharpness (3000 + 300 = 3300 dimensions)
        self.observation_space = my_gym.Box(
            low=np.zeros((3300,), dtype=np.float32), 
            high=np.ones((3300,), dtype=np.float32), 
            dtype=np.float32
        )
        
        self.ground_truth = np.random.uniform(self.min_val, self.max_val)
        self.max_steps = 10
        self.current_step = 0
        self.target_step = np.random.uniform(self.min_val, self.max_val)
        self.diff_threshold = (self.max_val - self.min_val) / self.max_steps

        self.meta_path = META_PATH
        self.lmdb_path = LMDB_PATH
        
        self._init_data()

    def _init_data(self):
        # Load Metadata
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata not found: {self.meta_path}")
            
        with open(self.meta_path, 'rb') as f:
            meta_info = pickle.load(f)
            
        self.label_to_data = {i: [] for i in range(10)}
        
        count = 0
        for item in meta_info:
            if len(item) == 3:
                key, label, sharpness_grid = item
                self.label_to_data[int(label)].append((sharpness_grid, key))
                count += 1
            else:
                pass
        
        self.total_entries = count
        print(f"Loaded metadata for {self.total_entries} images.")

        # Open LMDB
        if not os.path.exists(self.lmdb_path):
             raise FileNotFoundError(f"LMDB not found: {self.lmdb_path}")

        print("Loading LMDB into RAM..." )
        env_lmdb = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        self.key_to_tensor = {}
        
        with env_lmdb.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                key_str = key.decode('ascii')
                tensor_np = np.frombuffer(value, dtype=np.float32).reshape(10, 15, 20)
                self.key_to_tensor[key_str] = torch.from_numpy(tensor_np)
                
        env_lmdb.close()
        print(f"Loaded {len(self.key_to_tensor)} tensors from LMDB.")
        
        self.device = torch.device('cpu') 
        self.class_values = torch.arange(0, 10, dtype=torch.float32).unsqueeze(0)

    def _get_random_data_for_label(self, label):
        data_list = self.label_to_data[int(label)]
            
        assert hasattr(self, 'current_img_index'), "current_img_index not set"
        idx = self.current_img_index % len(data_list)
        sharpness_grid, key = data_list[idx]
        tensor = self.key_to_tensor.get(key, torch.zeros((10, 15, 20), dtype=torch.float32))
        return tensor, sharpness_grid

    def _compute_expected_radius_from_tensor(self, tensor):
        # Top-2 Soft Expectation logic from Project A
        
        # [10, 15, 20] -> [1, 10, 15, 20]
        input_tensor = tensor.unsqueeze(0) 
        logits_avg = input_tensor.mean(dim=(2, 3))
        
        # 1. Probabilities
        probs = torch.softmax(logits_avg, dim=1)
        
        # 2. Top-2 Masking
        k = 2
        topk_vals, topk_indices = torch.topk(probs, k, dim=1)
        
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk_indices, 1.0)
        
        masked_probs = probs * mask
        
        # 3. Renormalize
        masked_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-9)
        
        # 4. Expectation
        expected_radius = (masked_probs * self.class_values).sum(dim=1).item()
        return expected_radius

    def _get_interpolated_obs(self, val):
        val = np.clip(val, 0.0, 9.999)
        label_floor = int(np.floor(val))
        label_ceil = label_floor + 1
        
        weight_ceil = val - label_floor
        weight_floor = 1.0 - weight_ceil
        
        # Clamp labels
        label_floor = min(max(label_floor, 0), 9)
        label_ceil = min(max(label_ceil, 0), 9)
        
        t_floor, s_floor = self._get_random_data_for_label(label_floor)
        t_ceil, s_ceil = self._get_random_data_for_label(label_ceil)
        
        # Weighted Average
        t_mix = t_floor * weight_floor + t_ceil * weight_ceil
        radius_obs = val / 10.0 # Normalize to [0, 1]

        s_mix = s_floor * weight_floor + s_ceil * weight_ceil
        # We need the global scalar mean to feed the Old Expert MLP in the 'info' dict
        scalar_sharpness = np.mean(s_mix) / 640.0 / 480.0
        
        # Scale the scalar for the expert
        info_sharpness = scalar_sharpness * self.sharpness_scale

        # The Transformer needs the FULL 15x20 matrix, normalized similarly
        spatial_sharpness_obs = (s_mix / 640.0 / 480.0) * self.sharpness_scale
        
        # Observation: [11 channels, 15 height, 20 width] flattened
        stacked = np.concatenate([spatial_sharpness_obs[np.newaxis, :, :], t_mix.numpy()], axis=0)
        obs_tensor = stacked.flatten().astype(np.float32)
        
        return obs_tensor, {"sharpness": info_sharpness, "expected_radius": radius_obs}

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
        self.prev_position = self.target_step
        self.first_trial = True
        '''
        FSM states:
        - coarse search
        - fine search
        '''
        self.fsm = "coarse search"
        self.fsm_overshoot_count = 0
        obs, info = self._get_interpolated_obs(self.prev_diff * 10.0)
        return obs, info

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
        obs, info = self._get_interpolated_obs(absolute_diff * 10.0)
        
        self.current_step += 1
        improvement = self.prev_diff - absolute_diff
        r_guess = 0.0
        r_trigger = 0.0
        if self.first_trial:
            r_guess = 0.1
            r_trigger = 0.1 if action else -10.0
        else:
            sign = np.sign(self.prev_position - self.ground_truth) * np.sign(guess - self.ground_truth)
            if sign < 0:
                self.fsm_overshoot_count += 1
            if self.fsm == "coarse search":
                # calculate overshoot
                if self.fsm_overshoot_count == 1:
                    self.fsm = "fine search"
                else:
                    self.fsm = "coarse search"

                if action:
                    r_trigger = 0.1
                else:
                    r_trigger = -0.1

                if sign < 0:
                    r_guess = 1.0
                else:
                    r_guess = -absolute_diff
            else:
                # fine search
                #r_guess = -np.pow(np.clip(self.fsm_overshoot_count-2, a_min=0, a_max=10), 3)
                r_guess += -self.current_step * self.current_step / 10.0
                r_guess += improvement * 10.0
                if ((action == False) and (self.prev_diff < self.diff_threshold)):
                    r_trigger = 2.0
                elif (action == True) and (self.prev_diff < self.diff_threshold):
                    r_trigger = -2.0
                elif (action == False) and (self.prev_diff > self.diff_threshold):
                    r_trigger = -2.0
                elif (action == True) and (self.prev_diff > self.diff_threshold):
                    r_trigger = 1.0

        terminated = False
        if (self.first_trial == False) and (absolute_diff < self.diff_threshold) and (action == False):
            terminated = True
            r_guess = 10.0
            r_trigger = 10.0

        if self.current_step >= self.max_steps:
            terminated = True
            r_guess = -10.0
            r_trigger = -10.0

        self.prev_diff = absolute_diff
        
        total_reward = np.array([r_guess, r_trigger], dtype=np.float32)
        self.prev_position = guess
        self.first_trial = False
        return obs, total_reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            # 400x400 empty canvas
            return np.zeros((400, 400, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            pass
            
    def close(self):
        pass
