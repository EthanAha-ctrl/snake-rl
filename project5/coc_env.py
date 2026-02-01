import numpy as np
import my_gym

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
            low=np.array([0.0], dtype=np.float32), 
            high=np.array([1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # Observation space: Absolute difference in [0, 1]
        self.observation_space = my_gym.Box(
            low=np.array([0.0], dtype=np.float32), 
            high=np.array([1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        self.ground_truth = 0.0
        self.max_steps = 10
        self.current_step = 0
        self.diff_threshold = (self.max_val - self.min_val) / self.max_steps

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.ground_truth = np.random.uniform(self.min_val, self.max_val)
        self.current_step = 0
        initial_diff = 0
        self.prev_diff = float(initial_diff)
        return np.array([initial_diff], dtype=np.float32), {}

    def step(self, action): 
        # Action is expected to be a float or array of 1 float
        if isinstance(action, np.ndarray):
            guess = float(action.item())
        else:
            guess = float(action)
            
        guess = max(0.0, min(1.0, guess))
        absolute_diff = abs(guess - self.ground_truth)
        
        self.current_step += 1
        
        reward = 0.0
        terminated = False
        
        improvement = self.prev_diff - absolute_diff
        reward = 1 if improvement > 0 else -1
        reward -= (self.current_step * self.current_step) / 10.0
        
        if absolute_diff < self.diff_threshold:
            terminated = True
            reward += 10.0 # Bonus for success
        elif self.current_step >= self.max_steps:
            terminated = True
            reward -= 10.0 # Bonus for success
            
        self.prev_diff = absolute_diff
            
        return np.array([absolute_diff], dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            # 400x400 empty canvas
            return np.zeros((400, 400, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            pass
            
    def close(self):
        pass
