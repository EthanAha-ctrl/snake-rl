import numpy as np
import my_gym
import cv2
import os

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
            low=0.0, 
            high=1.0, 
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space: Absolute difference in [0, 1]
        self.observation_space = my_gym.Box(
            low=0.0, 
            high=1.0, 
            shape=(1,),
            dtype=np.float32
        )
        
        self.ground_truth = 0.0
        self.max_steps = 10
        self.current_step = 0
        self.diff_threshold = (self.max_val - self.min_val) / self.max_steps
        
        # Load background image
        image_path = os.path.join(os.path.dirname(__file__), "background.png")
        self.image = cv2.imread(image_path)
        if self.image is None:
            # Fallback to white noise if image not found, though user promised it exists
            self.image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        else:
            self.image = cv2.resize(self.image, (400, 400))
        self.last_diff = 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.ground_truth = np.random.uniform(self.min_val, self.max_val)
        self.current_step = 0
        initial_diff = 0
        self.prev_diff = float(initial_diff)
        self.last_diff = float(initial_diff)
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
        self.last_diff = absolute_diff
            
        return np.array([absolute_diff], dtype=np.float32), reward, terminated, False, {}

    def render(self):
        # Determine radius
        radius = self.last_diff * 10
        
        if radius < 0.5:
            # No blur effectively
            convolved_image = self.image.copy()
        else:
            # Generate AA kernel
            scale = 8
            aa_radius = radius * scale
            kernel_size_aa = int(2 * aa_radius) + 1
            # Ensure kernel size is odd
            if kernel_size_aa % 2 == 0:
                kernel_size_aa += 1
            
            # Create high-res kernel mask
            kernel_aa = np.zeros((kernel_size_aa, kernel_size_aa), dtype=np.float32)
            center = kernel_size_aa // 2
            cv2.circle(kernel_aa, (center, center), int(aa_radius), 1.0, -1)
            
            # Downscale kernel
            target_ksize = int(2 * radius) + 1
            if target_ksize % 2 == 0:
                target_ksize += 1
                
            kernel = cv2.resize(kernel_aa, (target_ksize, target_ksize), interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize
            kernel_sum = np.sum(kernel)
            if kernel_sum > 1e-6:
                kernel /= kernel_sum
                
            # Convolve
            convolved_image = cv2.filter2D(self.image, -1, kernel)

        if self.render_mode == "rgb_array":
            return convolved_image
        elif self.render_mode == "human":
            cv2.imshow("CoC Environment", convolved_image)
            cv2.waitKey(1)
            
    def close(self):
        pass
