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
    
    # Load the perfectly pre-trained BC weights
    try:
        trainer.load("sac_transformer_best.pth")
        print("Loaded BC pre-trained weights from sac_transformer_best.pth successfully.")
    except Exception as e:
        print(f"Failed to load BC weights: {e}")
        
    if torch.cuda.is_available() and config.phase == "RL_A":
        try:
            from torch import _dynamo
            _dynamo.config.suppress_errors = True
            trainer.encoder = torch.compile(trainer.encoder)
            print("Successfully compiled Transformer Encoder for speedup.")
        except Exception as e:
            print(f"Skipping torch.compile: {e}")
    
    # --- RL Phase A Mode ---
    print(f"Running in {config.phase} Mode.")
    if config.phase == "RL_A":
        print("The Transformer Encoder is FROZEN. The Actor/Critic MLPs are UNFROZEN.")
    
    trainingState = TrainingState()
    
    best_score = -float('inf')

    # SAC Loop: Step by Step (Off-Policy)
    from sac_trainer import Logger
    logger = Logger()
    
    reset_obs, info = env.reset()
    stacker.reset(reset_obs, default_obs=-1.0, default_action=-1.0)
    episode_transitions = []

    while trainingState.global_step < config.total_steps:
        # 1. Select Action
        current_stacked_obs = stacker.stacked()
        
        if trainingState.global_step < config.start_steps:
            # Random exploration
            action = env.action_space.sample()
        else:
            action = trainer.select_action(current_stacked_obs, evaluate=False)
            
        # 2. Step Env
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        
        trainingState.global_step += 1
        trainingState.update_episode_stats(reward, done)
        
        stacker.append(next_obs, action) 
        next_stacked_obs = stacker.stacked()

        episode_transitions.append((current_stacked_obs, action, reward, next_stacked_obs, done))
        
        if done:
            # --- 混合采样逻辑 (Mixed Sampling) ---
            keep_episode = True
            # 如果 episode 长度 < 5，有 90% 的概率丢弃它，只保留 10% 的短平快经验
            if len(episode_transitions) < 5 and trainingState.global_step >= config.start_steps:
                if np.random.rand() > 0.1:  # 90% probability to discard
                    keep_episode = False
            
            if keep_episode:
                for transition in episode_transitions:
                    trainer.replay_buffer.add(*transition)
            
            # Reset episode variables
            episode_transitions = []
            
            reset_obs, _ = env.reset()
            stacker.reset(reset_obs, default_obs=-1.0, default_action=-1.0)
            
        if trainingState.global_step >= config.start_steps:
            for _ in range(config.updates_per_step):
                losses = trainer.update_parameters(config.batch_size)
                
            if trainingState.global_step % 250 == 0:
                log_stats = trainingState.get_log_stats()
                log_stats.update(losses)
                
                # Custom Score: (r_guess + r_trigger) - mean_ep_len
                # mean_ep_reward already sums up (r_guess + r_trigger)
                # However, ep_rew_mean might be an array [r_guess_mean, r_trigger_mean]
                # We need to sum them up to get a scalar score.
                score = np.sum(log_stats["ep_rew_mean"]) - log_stats["ep_len_mean"]
                log_stats["score"] = score
                
                if score > best_score:
                    best_score = score
                    # Save best model
                    best_path = config.save_path.replace(".pth", "_best.pth")
                    trainer.save(best_path)
                    print(f"New best score: {score:.4f}. Saved model to {best_path}")

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
        batch_size=512,
        buffer_size=100_000,
        save_path="sac_coc.pth",
        phase="RL_A",
    )
    train(config)
