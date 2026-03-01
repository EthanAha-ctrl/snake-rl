import time
from typing import List, Dict, Union
import my_gym
import numpy as np
import os
import sys
import torch

# Switch to SAC
from sac_trainer import SACConfig, SACTrainer
from history_stacker import HistoryStacker
from coc_sharpness_env import CoCEnv


def make_env():
    return CoCEnv(render_mode="rgb_array")

def collect_expert_rollouts(env, new_trainer, num_episodes=500):
    #print(f"Collecting {num_episodes} expert rollouts into Replay Buffer...")
    
    # Dynamically load the old MLP Actor
    sys.path.insert(0, os.path.join("reference", "projectc"))
    from sac_trainer import Actor as OldActor
    sys.path.pop(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Old model inputs: obs_dim=2 (sharpness, radius), action_dim=2, history_len=10 -> total 40
    expert_actor = OldActor(obs_dim=40, action_dim=2, hidden_dim=256).to(device)
    checkpoint_path = os.path.join("reference", "projectc", "mlp_best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        expert_actor.load_state_dict(checkpoint['actor'])
        #print(f"Expert Actor loaded from {checkpoint_path}")
    else:
        pass
        #print(f"Warning: Expert checkpoint not found at {checkpoint_path}.")
        
    expert_actor.eval()
    
    old_stacker = HistoryStacker(obs_dim=2, action_dim=2, history_len=10)
    # The new stacker's dim will be inferred automatically inside `train.py`, so we can extract it:
    new_obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else 1
    new_action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1
    new_stacker = HistoryStacker(obs_dim=new_obs_dim, action_dim=new_action_dim, history_len=10)
    
    success_count = 0
    total_transitions = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        expert_obs = np.array([info.get("sharpness", 0.0), info.get("expected_radius", 0.0)], dtype=np.float32)
        
        old_stacker.reset(expert_obs, default_obs=-1.0, default_action=-1.0)
        new_stacker.reset(obs, default_obs=-1.0, default_action=-1.0)
        
        terminated = False
        truncated = False
        
        episode_transitions = []
        episode_reward = 0.0
        
        while not (terminated or truncated):
            expert_stacked = old_stacker.stacked()
            new_stacked = new_stacker.stacked()
            
            expert_stacked_t = torch.as_tensor(expert_stacked, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, std, logits = expert_actor(expert_stacked_t)
                guess = torch.sigmoid(mean)
                trigger_idx = torch.argmax(logits, dim=1)
                trigger_val = trigger_idx.float().unsqueeze(1)
                action_t = torch.cat([guess, trigger_val], dim=1)
                action = action_t.cpu().numpy()[0]
                
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += np.sum(reward)
            
            next_expert_obs = np.array([info.get("sharpness", 0.0), info.get("expected_radius", 0.0)], dtype=np.float32)
            
            old_stacker.append(next_expert_obs, action)
            new_stacker.append(next_obs, action)
            
            next_new_stacked = new_stacker.stacked()
            episode_transitions.append((new_stacked, action, reward, next_new_stacked, done))
            
        # Biased Sampling: Only save the trajectory if the expert succeeded (positive reward)
        if episode_reward > 0.0:
            success_count += 1
            for transition in episode_transitions:
                new_trainer.replay_buffer.add(*transition)
                total_transitions += 1
                
    #print(f"Expert collection finished. {success_count}/{num_episodes} successful episodes.")
    #print(f"Added {total_transitions} high-quality transitions to the Replay Buffer.")

class TrainingState:
    def __init__(self):
        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_ep_reward = 0.0
        self.current_ep_length = 0
        self.start_time = time.time()
        self.iterations = 0

    def update_episode_stats(self, reward: float, done: bool):
        self.current_ep_reward += reward
        self.current_ep_length += 1
        if done:
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)
            self.current_ep_reward = 0.0
            self.current_ep_length = 0

    def get_log_stats(self) -> Dict[str, float]:
        mean_ep_reward = (sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10)) if self.episode_rewards else 0.0
        mean_ep_len = (sum(self.episode_lengths[-10:]) / min(len(self.episode_lengths), 10)) if self.episode_lengths else 0.0
        time_elapsed = time.time() - self.start_time
        fps = int(self.global_step / time_elapsed) if time_elapsed > 0 else 0

        return {
            "ep_len_mean": mean_ep_len,
            "ep_rew_mean": mean_ep_reward,
            "fps": fps,
            "iterations": self.iterations,
            "time_elapsed": time_elapsed,
            "total_timesteps": self.global_step,
        }

def train(config: SACConfig = None):
    env = make_env()

    obs, _ = env.reset()
    
    # Obs dim for CoC is 1 (Box(1))
    obs_dim = 1 
    if hasattr(env.observation_space, 'shape'):
        obs_dim = env.observation_space.shape[0]

    # CoC Action dim is likely 2 now
    action_dim = 1
    if hasattr(env.action_space, 'shape'):
         action_dim = env.action_space.shape[0]

    stacker = HistoryStacker(obs_dim=obs_dim, action_dim=action_dim, history_len=config.history_len)
    stacker.reset(obs, default_obs=-1.0, default_action=-1.0)
    
    # Obs Dim in Buffer/Network = Stacked Dim
    # (obs_dim + action_dim) * history_len
    
    input_dim = obs_dim * config.history_len + action_dim * config.history_len
    
    trainer = SACTrainer(config, obs_dim=input_dim, action_dim=action_dim)
    
    # --- Pure Expert LfD Mode ---
    print("WARNING: Running in PURE EXPERT LfD Mode. The new model will NOT explore the environment.")
    print("All trajectories will be generated by the Old MLP (mlp_best.pth).")
    
    trainingState = TrainingState()
    best_score = -float('inf')
    
    from sac_trainer import Logger
    logger = Logger()
    
    while trainingState.global_step < config.total_steps:
        # 1. Collect a batch of Expert Data
        # Collect e.g., 5 expert episodes before each training cycle
        collect_expert_rollouts(env, trainer, num_episodes=5)
        
        # We need to manually increase global step to simulate time passing 
        # for logging purposes, assuming ~10 steps per episode
        new_steps = 5 * 10
        trainingState.global_step += new_steps
        
        # 2. Train on the full Buffer
        if trainer.replay_buffer.size > config.batch_size:
            # Train intensely on the new data using Behavior Cloning
            for _ in range(new_steps): # Update 1 time per step collected
                losses = trainer.update_bc(config.batch_size)
                
            if trainingState.global_step % 250 < 50: # Log roughly every 250 steps (approx 5 loops)
                log_stats = trainingState.get_log_stats()
                log_stats.update(losses)
                
                # We don't have real "student" rewards or SAC losses, 
                # so the score indicates the Behavior Cloning loss minimization
                score = -losses["loss_bc"] 
                log_stats["score"] = score
                
                if score > best_score:
                    best_score = score
                    best_path = config.save_path.replace(".pth", "_best.pth")
                    trainer.save(best_path)
                    print(f"New best internal score (minimized loss): {score:.4f}. Saved to {best_path}")

                logger.log(log_stats)
            
    trainer.save()
    env.close()


if __name__ == "__main__":
    config = SACConfig(
        total_steps=1_500_000,
        start_steps=5000, # Warmup random
        history_len=10,
        lr=2e-4,
        gamma=0.1,
        tau=0.005,
        alpha=0.05,
        hidden_dim=256,
        batch_size=32,
        buffer_size=20_000,
        save_path="sac_coc.pth",
    )
    train(config)
