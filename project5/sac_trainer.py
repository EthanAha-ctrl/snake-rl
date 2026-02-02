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
        self.rewards = torch.zeros((capacity, 2), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, obs, action, reward, next_obs, done):
        # inputs are numpy or scalar
        self.obs[self.ptr] = torch.as_tensor(obs, device=self.device)
        self.next_obs[self.ptr] = torch.as_tensor(next_obs, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, device=self.device)
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
        # We use Double Q-Learning (Two independent Critics: Q1 and Q2)
        
        # --- Q1 Network ---
        self.shared_net_1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        # Tower 1: Guess Q-Value
        self.guess_net_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Scalar Q_guess
        )
        # Tower 2: Trigger Q-Value
        self.trigger_net_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Scalar Q_trigger
        )

        # --- Q2 Network ---
        self.shared_net_2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        # Tower 1: Guess Q-Value
        self.guess_net_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Tower 2: Trigger Q-Value
        self.trigger_net_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        xu = torch.cat([obs, action], dim=1)
        
        # Q1 Forward
        feat1 = self.shared_net_1(xu)
        q1_guess = self.guess_net_1(feat1)
        q1_trigger = self.trigger_net_1(feat1)
        q1 = torch.cat([q1_guess, q1_trigger], dim=1) # (B, 2)

        # Q2 Forward
        feat2 = self.shared_net_2(xu)
        q2_guess = self.guess_net_2(feat2)
        q2_trigger = self.trigger_net_2(feat2)
        q2 = torch.cat([q2_guess, q2_trigger], dim=1) # (B, 2)
        
        return q1, q2

# Actor: pi(s) -> action distribution (Mean, Std)
# We use TanhGaussianPolicy
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        
        # 1. Shared Base (Low-level feature extraction)
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 2. Continuous Branch (Guess) - Dedicated MLP Tower
        self.guess_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.guess_mean = nn.Linear(hidden_dim, 1)
        self.guess_log_std = nn.Linear(hidden_dim, 1)
        
        # 3. Discrete Branch (Trigger) - Dedicated MLP Tower
        self.trigger_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.trigger_logits = nn.Linear(hidden_dim, 2)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        # Shared features
        shared_feat = self.shared_net(obs)
        
        # Continuous Path
        x_guess = self.guess_net(shared_feat)
        mean = self.guess_mean(x_guess)
        log_std = self.guess_log_std(x_guess)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Discrete Path
        x_trigger = self.trigger_net(shared_feat)
        logits = self.trigger_logits(x_trigger)
        
        return mean, std, logits
        
    def sample(self, obs):
        mean, std, logits = self(obs)
        
        # --- 1. Continuous Handling (Sigmoid Flow) ---
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization
        
        # Apply Sigmoid to squash to (0, 1) instead of Tanh
        guess = torch.sigmoid(x_t)
        
        # Log Prob correction for Sigmoid:
        # log_prob(y) = log_prob(x) - log(dy/dx)
        # dy/dx for sigmoid(x) is sigmoid(x)*(1-sigmoid(x)) = y*(1-y)
        # log(dy/dx) = log(y) + log(1-y)
        log_prob_cont = normal.log_prob(x_t)
        log_prob_cont -= torch.log(guess * (1.0 - guess) + 1e-6)
        log_prob_cont = log_prob_cont.sum(1, keepdim=True)
        
        # --- 2. Discrete Handling (Gumbel Softmax) ---
        # Gumbel-Softmax returns a differentiable ONE-HOT approximator
        # shape: (batch, 2)
        trigger_one_hot = F.gumbel_softmax(logits, tau=1.0, hard=False)
        
        # We need the scalar value for the 2nd dimension of action (the 'True' probability)
        # The env interprets value > 0.5 as True.
        # trigger_one_hot[:, 1] is the probability of being True.
        trigger_val = trigger_one_hot[:, 1].unsqueeze(1)
        
        # Log Prob for Discrete
        # For SAC, we usually use the log probability of the sampled class.
        # With Gumbel Softmax, we can use the Softmax probability.
        probs = F.softmax(logits, dim=-1)
        # We approximate log_pi as sum(prob * log_prob) (Entropy) or just log(prob_selected)
        # To keep it consistent with "sampled action", let's use the log prob of the class distribution
        # However, since trigger_val is continuous (relaxed), exact log_prob is tricky.
        # Standard Hybrid SAC approach: treat discrete part as minimizing KL (entropy)
        # log_pi = log(p_i).
        # We calculate entropy term: sum(p * log(p))
        log_prob_disc = (probs * torch.log(probs + 1e-6)).sum(dim=1, keepdim=True)
        
        # --- Combine ---
        action = torch.cat([guess, trigger_val], dim=1)
        
        # Total log_prob = log_prob_cont + log_prob_disc
        # (Assuming independence)
        log_prob = log_prob_cont + log_prob_disc
        
        return action, log_prob, action # returning 'action' as 'mean' tuple for compatibility if needed


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
            if evaluate:
                # Deterministic behavior
                mean, std, logits = self.actor(obs)
                
                # Continuous: just mean -> sigmoid (no sampling)
                # Wait, 'mean' is the logic before Sigma. 
                # But our distribution is Normal(mean, std).
                # If we want mode: mean.
                # Then we apply Sigmoid to Squash.
                # WAIT: Is mean pre-sigmoid? Yes.
                guess = torch.sigmoid(mean)
                
                # Discrete: Argmax
                # logits shape (1, 2). 
                trigger_idx = torch.argmax(logits, dim=1)
                trigger_val = trigger_idx.float().unsqueeze(1)
                
                action = torch.cat([guess, trigger_val], dim=1)
                
            else:
                # Stochastic behavior (Training)
                action, _, _ = self.actor.sample(obs)
        
        return action.cpu().numpy()[0] # (action_dim,)

    def update_parameters(self, batch_size):
        # Sample a batch from memory
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Target Actions
            next_state_actions, next_state_log_pi, _ = self.actor.sample(next_obs)
            
            q1_next_target, q2_next_target = self.critic_target(next_obs, next_state_actions)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) 
            
            # Entropy subtraction (Broadcasting scalar log_pi to vector Q)
            min_q_next_target = min_q_next_target - self.log_alpha.exp() * next_state_log_pi
            
            # Next Q Value: r + gamma * (Target)
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
        
        # Sum vector Qs -> Scalar Q for optimization
        total_q_pi = min_q_pi.sum(dim=1, keepdim=True)
        
        policy_loss = ((self.log_alpha.exp() * log_pi) - total_q_pi).mean()
        
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
             # Check if it is a Tensor or Numpy Array
             is_tensor = hasattr(v, 'cpu') # torch tensor
             is_numpy = isinstance(v, np.ndarray)
             
             size = 1
             if is_tensor:
                 size = v.numel()
                 if size > 1: v = v.cpu()
             elif is_numpy:
                 size = v.size
             
             if size > 1:
                 # Vector handling
                 v_arr = np.array(v).flatten()
                 for i, val in enumerate(v_arr):
                     print(f"|    {k}_{i:<18} | {val:<12.4f}|")
             else:
                 # Scalar handling
                 if hasattr(v, 'item'):
                     val = v.item()
                 else:
                     val = v
                 print(f"|    {k:<20} | {val:<12.4f}|")
        print("------------------------------------------")
