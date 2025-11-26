import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from torch import nn


# Only use position and angle from the raw CartPole observation
def select_obs(obs: np.ndarray) -> np.ndarray:
    return np.array([obs[0], obs[2]], dtype=np.float32)


class HistoryStacker:
    def __init__(self, obs_dim: int, history_len: int):
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.obs_history: Deque[np.ndarray] = deque(maxlen=history_len)
        self.action_history: Deque[float] = deque(maxlen=history_len)

    def reset(self, initial_obs: np.ndarray):
        self.obs_history.clear()
        self.action_history.clear()
        zero_obs = np.zeros(self.obs_dim, dtype=np.float32)
        for _ in range(self.history_len - 1):
            self.obs_history.append(zero_obs)
            self.action_history.append(0.0)
        self.obs_history.append(initial_obs.astype(np.float32))
        self.action_history.append(0.0)

    def append(self, obs: np.ndarray, action: float):
        self.obs_history.append(obs.astype(np.float32))
        self.action_history.append(float(action))

    def stacked(self) -> np.ndarray:
        obs_stack = np.concatenate(list(self.obs_history), dtype=np.float32)
        act_stack = np.array(self.action_history, dtype=np.float32)
        return np.concatenate([obs_stack, act_stack], dtype=np.float32)


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value.squeeze(-1)


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    log_prob: float
    reward: float
    value: float
    done: bool


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    return 1 - np.var(y_true - y_pred) / var_y if var_y > 1e-8 else 0.0


def compute_gae(transitions: List[Transition], gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    advantages = []
    gae = 0.0
    values = [t.value for t in transitions] + [0.0]
    for i in reversed(range(len(transitions))):
        delta = transitions[i].reward + gamma * values[i + 1] * (1.0 - transitions[i].done) - values[i]
        gae = delta + gamma * lam * (1.0 - transitions[i].done) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)


def make_env():
    env = gym.make("CartPole-v1")
    env.unwrapped.theta_threshold_radians = np.deg2rad(45.0)
    return env


@dataclass
class TrainConfig:
    total_steps: int = 200_000
    rollout_horizon: int = 2048
    history_len: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 10
    clip_epsilon: float = 0.2
    batch_size: int = 256
    hidden_dim: int = 128
    save_path: str = "ppo_cartpole.pth"


class Logger:
    def log(self, stats: Dict[str, float]) -> None:
        print("------------------------------------------")
        print("| rollout/                |              |")
        print(f"|    ep_len_mean          | {stats.get('ep_len_mean', 0):<12.0f}|")
        print(f"|    ep_rew_mean          | {stats.get('ep_rew_mean', 0):<12.0f}|")
        print("| time/                   |              |")
        print(f"|    fps                  | {int(stats.get('fps', 0)):<12d}|")
        print(f"|    iterations           | {int(stats.get('iterations', 0)):<12d}|")
        print(f"|    time_elapsed         | {int(stats.get('time_elapsed', 0)):<12d}|")
        print(f"|    total_timesteps      | {int(stats.get('total_timesteps', 0)):<12d}|")
        print("| train/                  |              |")
        print(f"|    approx_kl            | {stats.get('approx_kl', 0):<12.8f}|")
        print(f"|    clip_fraction        | {stats.get('clip_fraction', 0):<12.4f}|")
        print(f"|    clip_range           | {stats.get('clip_range', 0):<12.2f}|")
        print(f"|    entropy_loss         | {stats.get('entropy_loss', 0):<12.2f}|")
        print(f"|    explained_variance   | {stats.get('explained_variance', 0):<12.3f}|")
        print(f"|    learning_rate        | {stats.get('learning_rate', 0):<12.6f}|")
        print(f"|    loss                 | {stats.get('loss', 0):<12.4f}|")
        print(f"|    n_updates            | {int(stats.get('n_updates', 0)):<12d}|")
        print(f"|    policy_gradient_loss | {stats.get('policy_gradient_loss', 0):<12.4f}|")
        print(f"|    value_loss           | {stats.get('value_loss', 0):<12.4f}|")
        print("------------------------------------------")


class PPOTrainer:
    def __init__(self, config: TrainConfig, env=None):
        self.cfg = config
        self.env = env
        obs, _ = self.env.reset()
        obs = select_obs(obs)

        self.stacker = HistoryStacker(obs_dim=obs.shape[0], history_len=self.cfg.history_len)
        self.stacker.reset(obs)
        stacked_dim = obs.shape[0] * self.cfg.history_len + self.cfg.history_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(input_dim=stacked_dim, action_dim=self.env.action_space.n, hidden_dim=self.cfg.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        self.global_step = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_reward = 0.0
        self.episode_length = 0
        self.n_updates = 0
        self.iterations = 0
        self.logger = Logger()
        self.start_time = time.time()

    def collect_rollout(self) -> List[Transition]:
        transitions: List[Transition] = []
        for _ in range(self.cfg.rollout_horizon):
            stacked_obs = torch.as_tensor(self.stacker.stacked(), dtype=torch.float32, device=self.device)
            logits, value = self.model(stacked_obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            next_obs = select_obs(next_obs)

            self.episode_reward += reward
            self.episode_length += 1

            transitions.append(
                Transition(
                    obs=stacked_obs.cpu().numpy(),
                    action=action.item(),
                    log_prob=log_prob.item(),
                    reward=reward,
                    value=value.item(),
                    done=done,
                )
            )

            self.global_step += 1
            if done:
                self.episode_rewards.append(self.episode_reward)
                self.episode_lengths.append(self.episode_length)
                self.episode_reward = 0.0
                self.episode_length = 0
                next_obs, _ = self.env.reset()
                next_obs = select_obs(next_obs)
                self.stacker.reset(next_obs)
            else:
                self.stacker.append(next_obs, action.item())

            if self.global_step >= self.cfg.total_steps:
                break
        return transitions

    def update(self, transitions: List[Transition]) -> Dict[str, float]:
        advantages, returns = compute_gae(transitions, gamma=self.cfg.gamma, lam=self.cfg.gae_lambda)
        obs_batch = torch.as_tensor(np.stack([t.obs for t in transitions]), dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor([t.action for t in transitions], dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor([t.log_prob for t in transitions], dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        num_samples = len(transitions)
        pg_losses, value_losses, ent_losses, kl_list, clip_fracs, ev_list = [], [], [], [], [], []
        for _ in range(self.cfg.ppo_epochs):
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                mb_idx = indices[start:end]
                mb_obs = obs_batch[mb_idx]
                mb_actions = action_batch[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv = advantages_t[mb_idx]

                logits, values = self.model(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(mb_actions)

                ratio = (new_log_probs - mb_old_log_probs).exp()
                pg_loss = -torch.min(
                    ratio * mb_adv,
                    torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * mb_adv,
                ).mean()
                value_loss = nn.functional.mse_loss(values, mb_returns)
                loss = pg_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                ent_losses.append(entropy.item())
                kl_list.append((mb_old_log_probs - new_log_probs).mean().item())
                clip_fracs.append((ratio.gt(1 + self.cfg.clip_epsilon) | ratio.lt(1 - self.cfg.clip_epsilon)).float().mean().item())
                ev_list.append(
                    explained_variance(
                        y_pred=values.detach().cpu().numpy(),
                        y_true=mb_returns.detach().cpu().numpy(),
                    )
                )
                self.n_updates += 1

        return {
            "approx_kl": float(np.mean(kl_list)) if kl_list else 0.0,
            "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "entropy_loss": float(-np.mean(ent_losses)) if ent_losses else 0.0,
            "loss": float(np.mean(value_losses) + np.mean(pg_losses) - 0.01 * np.mean(ent_losses)) if pg_losses else 0.0,
            "policy_gradient_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "explained_variance": float(np.mean(ev_list)) if ev_list else 0.0,
        }

    def log(self, train_stats: Dict[str, float]) -> None:
        mean_ep_reward = (sum(self.episode_rewards[-10:]) / min(len(self.episode_rewards), 10)) if self.episode_rewards else 0.0
        mean_ep_len = (sum(self.episode_lengths[-10:]) / min(len(self.episode_lengths), 10)) if self.episode_lengths else 0.0
        time_elapsed = time.time() - self.start_time
        fps = int(self.global_step / time_elapsed) if time_elapsed > 0 else 0
        lr_now = self.optimizer.param_groups[0]["lr"]

        stats = {
            "ep_len_mean": mean_ep_len,
            "ep_rew_mean": mean_ep_reward,
            "fps": fps,
            "iterations": self.iterations,
            "time_elapsed": time_elapsed,
            "total_timesteps": self.global_step,
            "approx_kl": train_stats.get("approx_kl", 0.0),
            "clip_fraction": train_stats.get("clip_fraction", 0.0),
            "clip_range": self.cfg.clip_epsilon,
            "entropy_loss": train_stats.get("entropy_loss", 0.0),
            "explained_variance": train_stats.get("explained_variance", 0.0),
            "learning_rate": lr_now,
            "loss": train_stats.get("loss", 0.0),
            "n_updates": self.n_updates,
            "policy_gradient_loss": train_stats.get("policy_gradient_loss", 0.0),
            "value_loss": train_stats.get("value_loss", 0.0),
        }
        self.logger.log(stats)

    def save(self):
        torch.save(self.model.state_dict(), self.cfg.save_path)

    def run(self):
        while self.global_step < self.cfg.total_steps:
            self.iterations += 1
            transitions = self.collect_rollout()
            train_stats = self.update(transitions)
            self.log(train_stats)
        self.save()
        self.env.close()


def train(config: TrainConfig = TrainConfig()):
    trainer = PPOTrainer(config, env = make_env())
    trainer.run()


if __name__ == "__main__":
    train()
