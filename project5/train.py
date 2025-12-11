import time
from typing import List, Dict, Union
import my_gym
import numpy as np

# Switch to SAC
from sac_trainer import SACConfig, SACTrainer
from history_stacker import HistoryStacker, position_angle_from_obs
from coc_env import CoCEnv


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

    stacker = HistoryStacker(obs_dim=obs_dim, history_len=config.history_len)
    stacker.reset(obs, default_obs=-1.0, default_action=-1.0)
    
    # Obs Dim in Buffer/Network = Stacked Dim
    # (obs_dim + action_dim) * history_len
    # Wait, SACTrainer takes input_dim
    # CoC Action dim is 1
    action_dim = 1
    
    input_dim = obs_dim * config.history_len + config.history_len # 10 * (1+1) = 20
    
    trainer = SACTrainer(config, obs_dim=input_dim, action_dim=action_dim)
    trainingState = TrainingState()
    
    # SAC Loop: Step by Step (Off-Policy)
    from sac_trainer import Logger
    logger = Logger()
    
    while trainingState.global_step < config.total_steps:
        # 1. Select Action
        current_stacked_obs = stacker.stacked()
        
        if trainingState.global_step < config.start_steps:
            # Random exploration
            action = env.action_space.sample()
            # Ensure it's float numpy array if needed, but sample() usually correct
            # CoC action is Box(1), sample returns np.array([0.123])
            # If env.action_space is Discrete, we have problems. But CoC is Box.
            if isinstance(action, np.ndarray):
                action = action.astype(np.float32)
            else:
                action = np.array([action], dtype=np.float32)
        else:
            action = trainer.select_action(current_stacked_obs, evaluate=False)
            
        # 2. Step Env
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        trainingState.global_step += 1
        trainingState.update_episode_stats(reward, done)
        
        # 3. Handle Stacker for Next State
        # We need "next_stacked_obs" for buffer
        # This is tricky with history.
        # Temp append to get next stack, but we need to keep stacker consistent for next loop iter.
        # Actually stacker state evolves naturally.
        # Current Stacker: H_t
        # Action: a_t
        # Next Obs: o_{t+1}
        # Next Stacker should result in H_{t+1}
        
        # We perform append to get next stack
        # (We will need to copy purely for buffer? No, simple append is fine because buffer stores copy)
        # BUT: if done, we must handle next_obs correctly.
        # In SAC, 'next_state' for terminal state matters.
        # If not done: append (o_{t+1}, a_t).
        # If done: we still append it to form the "terminal next state", 
        # and THEN we reset stacker for the FUTURE steps.
        
        # NOTE: standard stacker.append updates internal state.
        # We need the state BEFORE update as 'current_obs' (we have it: current_stacked_obs)
        # We need state AFTER update as 'next_obs'
        
        if isinstance(action, np.ndarray) and action.ndim > 0:
            action_scalar = action.item()
        else:
            action_scalar = float(action)
            
        stacker.append(next_obs, action_scalar) 
        next_stacked_obs = stacker.stacked()
        
        # 4. Add to Buffer
        # Action in buffer: flat array
        trainer.replay_buffer.add(
            current_stacked_obs, 
            action, 
            reward, 
            next_stacked_obs, 
            done
        )
        
        # 5. Reset if Done
        if done:
            reset_obs, _ = env.reset()
            stacker.reset(reset_obs, default_obs=-1.0, default_action=-1.0)
            
        # 6. Update Parameters
        if trainingState.global_step >= config.start_steps:
            # Update multiple times per step? Usually 1
            for _ in range(config.updates_per_step):
                losses = trainer.update_parameters(config.batch_size)
                
            if trainingState.global_step % 1000 == 0:
                log_stats = trainingState.get_log_stats()
                log_stats.update(losses)
                logger.log(log_stats)

    trainer.save()
    env.close()


if __name__ == "__main__":
    config = SACConfig(
        total_steps=50_000,
        start_steps=5000, # Warmup random
        history_len=10,
        lr=3e-4,
        gamma=0.1,
        tau=0.005,
        alpha=0.2,
        hidden_dim=256,
        batch_size=256,
        buffer_size=100_000,
        save_path="sac_coc.pth",
    )
    train(config)
