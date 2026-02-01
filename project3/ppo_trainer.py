import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn




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


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
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


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    return 1 - np.var(y_true - y_pred) / var_y if var_y > 1e-8 else 0.0


def _compute_gae(transitions: List[Transition], gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    advantages = []
    gae = 0.0
    values = [t.value for t in transitions] + [0.0]
    for i in reversed(range(len(transitions))):
        delta = transitions[i].reward + gamma * values[i + 1] * (1.0 - transitions[i].done) - values[i]
        gae = delta + gamma * lam * (1.0 - transitions[i].done) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)


class Logger:
    def log(self, stats: Dict[str, float]) -> None:
        print("------------------------------------------")
        print("| rollout/                |             |")
        print(f"|    ep_len_mean          | {stats.get('ep_len_mean', 0):<12.0f}|")
        print(f"|    ep_rew_mean          | {stats.get('ep_rew_mean', 0):<12.0f}|")
        print("| time/                   |             |")
        print(f"|    fps                  | {int(stats.get('fps', 0)):<12d}|")
        print(f"|    iterations           | {int(stats.get('iterations', 0)):<12d}|")
        print(f"|    time_elapsed         | {int(stats.get('time_elapsed', 0)):<12d}|")
        print(f"|    total_timesteps      | {int(stats.get('total_timesteps', 0)):<12d}|")
        print("| train/                  |             |")
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
    def __init__(self, config: TrainConfig, input_dim: int, action_dim: int):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(input_dim=input_dim, action_dim=action_dim, hidden_dim=self.cfg.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.logger = Logger()
        self.n_updates = 0

    def predict(self, obs: np.ndarray) -> Tuple[int, float, float]:
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            logits, value = self.model(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def update_weights(self, transitions: List[Transition]) -> Dict[str, float]:
        advantages, returns = _compute_gae(transitions, gamma=self.cfg.gamma, lam=self.cfg.gae_lambda)
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
                    _explained_variance(
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

    def log(self, stats: Dict[str, float]) -> None:
        stats["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        stats["n_updates"] = self.n_updates
        self.logger.log(stats)

    def save(self):
        torch.save(self.model.state_dict(), self.cfg.save_path)
