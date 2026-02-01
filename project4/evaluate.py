import numpy as np
import torch
import gymnasium as gym
import cv2

from history_stacker import HistoryStacker, position_angle_from_obs
from ppo_trainer import ActorCritic
from vision_extractor import VisionStateExtractor

history_len = 4
render_mode = "rgb_array"
USE_VISION_EXTRACTOR = True

env = gym.make("CartPole-v1", render_mode=render_mode)
env.unwrapped.theta_threshold_radians = np.deg2rad(45.0)
env.unwrapped.x_threshold = 1.5

extractor = VisionStateExtractor(model_path="position_detection.pth") if USE_VISION_EXTRACTOR else None

state, _ = env.reset()
frame = env.render()
if USE_VISION_EXTRACTOR:
    initial_obs = extractor.predict(frame)
else:
    initial_obs = position_angle_from_obs(state)

stacker = HistoryStacker(obs_dim=initial_obs.shape[0], history_len=history_len)
stacker.reset(initial_obs)

stacked_dim = initial_obs.shape[0] * history_len + history_len
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(input_dim=stacked_dim, action_dim=env.action_space.n).to(device)
state_dict = torch.load("ppo_cartpole.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

episode_lengths = []
current_len = 0
n_episodes = 50

print(f"Evaluating for {n_episodes} episodes...")

while len(episode_lengths) < n_episodes:
    stacked_obs = torch.as_tensor(stacker.stacked(), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, _ = model(stacked_obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.probs.argmax().item()
    
    next_state, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    
    # cv2.imshow("CartPole-v1 (vision eval)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)
    
    if USE_VISION_EXTRACTOR:
        next_obs_vec = extractor.predict(frame)
    else:
        next_obs_vec = position_angle_from_obs(next_state)
        
    current_len += 1
    
    if terminated or truncated:
        episode_lengths.append(current_len)
        print(f"Episode {len(episode_lengths)}: Length {current_len}")
        current_len = 0
        
        reset_state, _ = env.reset()
        reset_frame = env.render()
        if USE_VISION_EXTRACTOR:
            stacker.reset(extractor.predict(reset_frame))
        else:
            stacker.reset(position_angle_from_obs(reset_state))
        continue
        
    stacker.append(next_obs_vec, action)

env.close()
# cv2.destroyAllWindows()

episode_lengths = np.array(episode_lengths)
print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
print(f"Max Episode Length: {np.max(episode_lengths)}")
print(f"Min Episode Length: {np.min(episode_lengths)}")
