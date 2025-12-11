import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict, List
import copy

class SACConfig:
    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2, # Entropy coefficient (can be learnable)
        hidden_dim: int = 256,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        updates_per_step: int = 1,
        start_steps: int = 1000, # Random exploration steps
        total_steps: int = 200_000,
        rollout_horizon: int = 1, # SAC is off-policy, horizon is 1 usually or meaningless
        history_len: int = 4,
        save_path: str = "sac_model.pth"
    ):
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.updates_per_step = updates_per_step
        self.start_steps = start_steps
        self.total_steps = total_steps
        self.rollout_horizon = rollout_horizon
        self.history_len = history_len
        self.save_path = save_path

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.device = device
        
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, obs, action, reward, next_obs, done):
        # inputs are numpy or scalar
        self.obs[self.ptr] = torch.as_tensor(obs, device=self.device)
        self.next_obs[self.ptr] = torch.as_tensor(next_obs, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, device=self.device)
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx]
        )

# Networks
# Critic: Q(s, a) -> value
class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        # We usually use Double Q-Learning in SAC (two critics)
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

    def forward(self, obs, action):
        xu = torch.cat([obs, action], dim=1)
        return self.q1(xu), self.q2(xu)

# Actor: pi(s) -> action distribution (Mean, Std)
# We use TanhGaussianPolicy
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        x = self.fc(obs)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std
        
    def sample(self, obs):
        mean, std = self(obs)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)   # Squash to [-1, 1]
        
        action = y_t
        # Enforce action bounds if needed? In this Env it's [0, 1].
        # Tanh outputs [-1, 1]. We need to scale to [0, 1].
        # scale: (action + 1) / 2
        action = (action + 1) / 2.0
        
        # Log prob calculation correction for Tanh squash
        # log_prob = log_prob_normal - sum(log(1 - tanh(x)^2))
        log_prob = normal.log_prob(x_t)
        # Correction formula: log(1 - tanh(x)^2 + epsilon)
        # But we also sticked a linear transform (x+1)/2 after tanh.
        # Let's derive: y = (tanh(x) + 1)/2. 
        # We need log_det_jacobian.
        # This is getting complex mathematically. 
        # Standard SAC assumes action in [-1, 1].
        # Easiest way: Let the Env wrapper handle scaling [-1, 1] -> [0, 1].
        # And keep Actor outputting [-1, 1].
        
        # WAIT: For this implementation, let's stick to standard SAC outputting [-1, 1].
        # And we perform unwrapping in the Trainer.predict or Environment Step.
        
        # Standard correction for Tanh
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, y_t # Return y_t (tanh output) as raw action [-1, 1]


class SACTrainer:
    def __init__(self, config: SACConfig, obs_dim: int, action_dim: int):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim, action_dim, self.device)
        
        # Networks
        self.critic = Critic(obs_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor = Actor(obs_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        
        self.q_optimizer = optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        
        # Automatic Entropy Tuning (Optional)
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.cfg.lr)

    def select_action(self, obs: np.ndarray, evaluate=False):
        # Obs: numpy (obs_dim,)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Force deterministic behavior as requested
            mean, std = self.actor(obs)
            action = torch.tanh(mean)
            action = (action + 1) / 2.0 # Scale [-1, 1] -> [0, 1]
        
        return action.cpu().numpy()[0] # (action_dim,)

    def update_parameters(self, batch_size):
        # Sample a batch from memory
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Target Actions
            next_state_actions, next_state_log_pi, _ = self.actor.sample(next_obs)
            # Input to Q-target needs to be raw [-1, 1] usually?
            # Wait, Critic takes (s, a). If 'a' in buffer is [0, 1], then Critic learns on [0, 1].
            # So next_state_actions must be [0, 1] too. 
            # My Actor.sample returns [0, 1]. So we are consistent.
            
            q1_next_target, q2_next_target = self.critic_target(next_obs, next_state_actions)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.log_alpha.exp() * next_state_log_pi
            next_q_value = rewards + (1 - dones) * self.cfg.gamma * min_q_next_target
            
        # Critic Update
        q1, q2 = self.critic(obs, actions)
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Actor Update
        pi, log_pi, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        policy_loss = ((self.log_alpha.exp() * log_pi) - min_q_pi).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Alpha Update
        alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft Update Targets
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.cfg.tau) + param.data * self.cfg.tau)
            
        return {
            "loss_q": q_loss.item(),
            "loss_pi": policy_loss.item(),
            "alpha": self.log_alpha.exp().item()
        }

    def save(self, path=None):
        path = path or self.cfg.save_path
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'log_alpha': self.log_alpha
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path=None):
        path = path or self.cfg.save_path
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.log_alpha = checkpoint['log_alpha']

class Logger:
    def log(self, stats: Dict[str, float]) -> None:
        print("------------------------------------------")
        print("| train/                  |             |")
        for k, v in stats.items():
             print(f"|    {k:<20} | {v:<12.4f}|")
        print("------------------------------------------")
