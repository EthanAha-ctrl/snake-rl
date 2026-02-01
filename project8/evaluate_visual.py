
import sys
import os
import cv2
import torch
import numpy as np

# Add reference path to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
reference_dir = os.path.join(current_dir, "reference")
sys.path.append(reference_dir)

# Import from reference directory
from reference.coc_env import CoCEnv
from reference.my_gym import Box
from reference.sac_trainer import SACTrainer, SACConfig
from reference.history_stacker import HistoryStacker
from reference.model import get_hrnet_w18

class VisionExtractor:
    def __init__(self, model_path, device):
        self.device = device
        print(f"Loading Vision Model from {model_path}...")
        self.model = get_hrnet_w18(num_classes=50, in_channels=1)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            self.model.eval()
            print("Vision Model loaded successfully.")
        else:
            raise FileNotFoundError(f"Vision checkpoint not found at {model_path}")

    def predict(self, image_bgr):
        """
        Input: BGR Image (H, W, 3) from Env render
        Output: Estimated absolute difference (float)
        """
        # 1. Convert to Grayscale
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        # 2. Normalize and transform to Tensor [1, 1, H, W]
        input_tensor = gray.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0).to(self.device)

        # 3. Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            # Output: [1, 50, H, W] dense prediction
            
            # Global Average of logits/probabilities
            # (Matches project7/evaluate.py logic: mean over spatial dims)
            logits_avg = output.mean(dim=(2, 3)) 
            
            # --- START: Soft Expectation (Idea 1.5) ---
            # Instead of argmax, we use the weighted average of probabilities
            probabilities = torch.softmax(logits_avg, dim=1) # [1, 50]
            
            # Create a tensor of class values [1.0, 2.0, ..., 50.0]
            # Since pred_idx 0 corresponds to Radius 1
            class_values = torch.arange(1, 51, device=self.device, dtype=torch.float32).unsqueeze(0)
            
            # Expected Radius = Sum(Prob * Value)
            expected_radius = (probabilities * class_values).sum(dim=1).item()
            # print(f"DEBUG: Soft Radius: {expected_radius:.4f}")
            
            pred_radius = expected_radius
            # --- END: Soft Expectation ---
        
        # 5. Convert Radius to Diff
        # In coc_env.py: radius = last_diff * 10
        # Therefore: last_diff = radius / 10.0
        estimated_diff = pred_radius / 10.0
        
        return np.array([estimated_diff], dtype=np.float32)

def main():
    # Configuration
    HISTORY_LEN = 10
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    SAC_MODEL_PATH = os.path.join(reference_dir, "sac_coc.pth")
    VISION_MODEL_PATH = os.path.join(reference_dir, "best_model.pth")
    
    # Initialize Environment
    # Note: coc_env needs background.png which is now in reference/
    # We might need to ensure coc_env looks in the right place or we run from reference dir
    env = CoCEnv(render_mode='rgb_array')
    
    # Initialize Vision Model
    try:
        extractor = VisionExtractor(VISION_MODEL_PATH, DEVICE)
    except FileNotFoundError as e:
        print(e)
        return

    # Initialize RL Agent (SAC)
    print(f"Loading SAC Agent from {SAC_MODEL_PATH}...")
    
    # Determine dims for SAC
    # CoC Obs Dim = 1, Action Dim = 1
    obs_dim = 1
    action_dim = 1
    
    # Input dim to SAC is the stacked vector size
    # shape = (obs_dim * history_len) + history_len
    input_dim = (obs_dim * HISTORY_LEN) + HISTORY_LEN
    
    config = SACConfig(
        history_len=HISTORY_LEN,
        save_path=SAC_MODEL_PATH,
        hidden_dim=256 # Assuming default from train.py
    )
    
    trainer = SACTrainer(config, obs_dim=input_dim, action_dim=action_dim)
    
    if os.path.exists(SAC_MODEL_PATH):
        trainer.load(SAC_MODEL_PATH)
        print("SAC Agent loaded successfully.")
    else:
        print(f"Warning: SAC model not found at {SAC_MODEL_PATH}. Running with random initialization.")

    # Initialize Stacker
    stacker = HistoryStacker(obs_dim=obs_dim, history_len=HISTORY_LEN)

    # Evaluation Loop
    n_episodes = 10
    total_rewards = []
    episode_lengths = []

    print(f"\nStarting Visual Evaluation for {n_episodes} episodes...")
    print("-------------------------------------------------------")

    for ep in range(n_episodes):
        _ = env.reset()
        
        frame = env.render() # Returns RGB Array (actually BGR if cv2 used underneath without conversion, check env)
        vision_obs = extractor.predict(frame)
        
        stacker.reset(vision_obs, default_obs=-1.0, default_action=-1.0)
        
        ep_reward = 0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            current_stacked_obs = stacker.stacked()
            action = trainer.select_action(current_stacked_obs, evaluate=True)
            
            next_real_obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()
            next_vision_obs = extractor.predict(frame)
            
            print(f"Step {steps}: Act={action[0]:.3f}, Real Diff={next_real_obs[0]:.3f}, Vis Diff={next_vision_obs[0]:.3f}")
            
            # IMPORTANT: We use action[0] because select_action returns array
            stacker.append(next_vision_obs, action[0])
            
            ep_reward += reward
            steps += 1
            
            cv2.imshow("Visual Agent Evaluation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                terminated = True # Force quit
                
        total_rewards.append(ep_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}: Steps={steps}, Reward={ep_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()
    
    avg_len = np.mean(episode_lengths)
    avg_rew = np.mean(total_rewards)
    print("-------------------------------------------------------")
    print(f"Evaluation Complete.")
    print(f"Average Steps: {avg_len:.2f}")
    print(f"Average Reward: {avg_rew:.2f}")

if __name__ == "__main__":
    main()
