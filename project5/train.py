import time
from typing import List, Dict, Union
import my_gym
import numpy as np

from ppo_trainer import TrainConfig, PPOTrainer, Transition
from history_stacker import HistoryStacker #, position_angle_from_obs # Removed unused import
from coc_env import CoCEnv


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

def make_env():
    return CoCEnv(render_mode="rgb_array")

def collect_rollout(
    env: my_gym.Env,
    stacker: HistoryStacker,
    trainer: PPOTrainer,
    config: TrainConfig,
    trainingState: TrainingState
) -> List[Transition]:
    transitions: List[Transition] = []

    for _ in range(config.rollout_horizon):
        obs_stack = stacker.stacked()
        action, log_prob, value = trainer.predict(obs_stack)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        transitions.append(
            Transition(
                obs=obs_stack,
                action=action,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=done,
            )
        )

        trainingState.global_step += 1
        trainingState.update_episode_stats(reward, done)

        if done:
            reset_obs, _ = env.reset()
            # CoC obs is 1D, just use it
            stacker.reset(reset_obs)
        else:
            # CoC obs is 1D, just use it
            stacker.append(next_obs, action)

        if trainingState.global_step >= config.total_steps:
            break

    return transitions


def train(config: TrainConfig = None):
    env = make_env()

    obs, _ = env.reset()
    # obs = position_angle_from_obs(obs) # Not needed for CoC
    
    # Obs dim for CoC is 1 (Box(1))
    # We rely on env.observation_space.shape[0] potentially, or just code it
    obs_dim = 1 
    if hasattr(env.observation_space, 'shape'):
        obs_dim = env.observation_space.shape[0]

    stacker = HistoryStacker(obs_dim=obs_dim, history_len=config.history_len)
    stacker.reset(obs)

    # Input dim: (Obs dim + Action dim) * history_len?
    # HistoryStacker stackes obs and action.
    # obs dim = 1
    # action dim = 1 (CoC is Box(1))
    
    # Wait, HistoryStacker appends action (float) to action_history.
    # stacked() returns obs_stack + act_stack.
    # obs_stack is history_len * obs_dim
    # act_stack is history_len * 1
    
    # Start PPOTrainer
    is_continuous = isinstance(env.action_space, my_gym.Box)
    action_dim = 1 if is_continuous else env.action_space.n
    if is_continuous and hasattr(env.action_space, 'shape'):
         action_dim = env.action_space.shape[0]

    input_dim = obs_dim * config.history_len + config.history_len # + history_len for actions

    trainer = PPOTrainer(config, input_dim=input_dim, action_dim=action_dim, is_continuous=is_continuous)
    trainingState = TrainingState()

    while trainingState.global_step < config.total_steps:
        trainingState.iterations += 1
        transitions = collect_rollout(env, stacker, trainer, config, trainingState)
        train_stats = trainer.update_weights(transitions)

        log_stats = trainingState.get_log_stats()
        log_stats.update(train_stats)
        trainer.log(log_stats)

    trainer.save()
    env.close()


if __name__ == "__main__":
    config = TrainConfig(
        total_steps=50_000,
        rollout_horizon=2048,
        history_len=4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=10,
        clip_epsilon=0.2,
        batch_size=256,
        hidden_dim=128,
        save_path="ppo_coc.pth",
    )
    train(config)
