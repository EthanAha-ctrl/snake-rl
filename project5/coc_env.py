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
        
        self.ground_truth = 0.0
        self.max_steps = 10
        self.current_step = 0
        self.target_step = 0
        self.diff_threshold = (self.max_val - self.min_val) / self.max_steps
        self.reached = False

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.ground_truth = np.random.uniform(self.min_val, self.max_val)
        self.current_step = 0
        self.target_step = 0
        self.prev_diff = abs(self.target_step - self.ground_truth)
        self.reached = False
        return np.array([self.prev_diff], dtype=np.float32), {}

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
                # once reached before
                if action == False:
                    r_trigger += 1.0
                    terminated = True
                else:
                    r_trigger = -10.0
            else:
                # just reached for the first time
                if action == False:
                    r_trigger -= (self.current_step * self.current_step)
                else:
                    r_trigger += 1.0

        self.prev_diff = absolute_diff
        
        total_reward = np.array([r_guess, r_trigger], dtype=np.float32)
        
        if absolute_diff < self.diff_threshold:
            self.reached = True

        return np.array([absolute_diff], dtype=np.float32), total_reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            # 400x400 empty canvas
            return np.zeros((400, 400, 3), dtype=np.uint8)
        elif self.render_mode == "human":
            pass
            
    def close(self):
        pass
