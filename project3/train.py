import time
import pickle
import os
from typing import List, Dict, Optional
import gymnasium as gym
import numpy as np

from ppo_trainer import TrainConfig, PPOTrainer, Transition
from history_stacker import HistoryStacker, position_angle_from_obs


def make_env():
    env = gym.make("CartPole-v1")
    env.unwrapped.theta_threshold_radians = np.deg2rad(45.0)
    env.unwrapped.x_threshold = 1.5
    return env


class NoiseInjector:
    def __init__(self, path="error_metrics.pkl"):
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Noise injection disabled.")
            self.active = False
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        pos_errors = np.array(data.get("pos_errors", []))
        angle_errors_deg = np.array(data.get("angle_errors_deg", []))

        if len(pos_errors) == 0:
            self.active = False
            return

        # Model as Multivariate Normal Distribution
        self.mean = np.array([np.mean(pos_errors), np.mean(angle_errors_deg)])
        self.cov = np.cov(pos_errors, angle_errors_deg)
        self.active = True
        print(f"Noise Injector initialized using {path}")
        print(f"Noise Mean: {self.mean}")
        print(f"Noise Covariance:\n{self.cov}")

    def add_noise(self, obs: np.ndarray) -> np.ndarray:
        # obs is [pos, angle_rad]
        if not self.active:
            return obs

        noise = np.random.multivariate_normal(self.mean, self.cov)
        # noise is [pos_err, angle_err_deg]

        noisy_obs = obs.copy()
        noisy_obs[0] += noise[0]
        noisy_obs[1] += np.deg2rad(noise[1])

        return noisy_obs


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


def collect_rollout(
    env: gym.Env,
    stacker: HistoryStacker,
    trainer: PPOTrainer,
    config: TrainConfig,
    trainingState: TrainingState,
    noise_injector: Optional[NoiseInjector] = None
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
            reset_obs_selected = position_angle_from_obs(reset_obs)
            if noise_injector:
                reset_obs_selected = noise_injector.add_noise(reset_obs_selected)
            stacker.reset(reset_obs_selected)
        else:
            next_obs_selected = position_angle_from_obs(next_obs)
            if noise_injector:
                next_obs_selected = noise_injector.add_noise(next_obs_selected)
            stacker.append(next_obs_selected, action)

        if trainingState.global_step >= config.total_steps:
            break

    return transitions


def train(config: TrainConfig = None):
    env = make_env()

    
    noise_injector = NoiseInjector("error_metrics.pkl")

    obs, _ = env.reset()
    obs = position_angle_from_obs(obs)
    obs = noise_injector.add_noise(obs)
    stacker = HistoryStacker(obs_dim=obs.shape[0], history_len=config.history_len)
    stacker.reset(obs)

    input_dim = obs.shape[0] * config.history_len + config.history_len
    action_dim = env.action_space.n

    trainer = PPOTrainer(config, input_dim=input_dim, action_dim=action_dim)
    trainingState = TrainingState()

    while trainingState.global_step < config.total_steps:
        trainingState.iterations += 1
        transitions = collect_rollout(env, stacker, trainer, config, trainingState, noise_injector)
        train_stats = trainer.update_weights(transitions)

        log_stats = trainingState.get_log_stats()
        log_stats.update(train_stats)
        trainer.log(log_stats)

    trainer.save()
    env.close()


if __name__ == "__main__":
    config = TrainConfig(
        total_steps=800_000,
        rollout_horizon=2048,
        history_len=4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=10,
        clip_epsilon=0.2,
        batch_size=256,
        hidden_dim=128,
        save_path="ppo_cartpole.pth",
    )
    train(config)
