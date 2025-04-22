import os
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# ========== Replay Buffers ============
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*transitions))

    def __len__(self):
        return len(self.buffer)

# Prioritized Experience Replay
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.buffer = [None] * capacity
        self.frame = 1

    def push(self, *transition):
        max_prio = self.priorities.max() if self.buffer[self.pos] is not None else 1.0
        self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = map(np.stack, zip(*samples))
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len([x for x in self.buffer if x is not None])

# ============ Networks ==============
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Dueling DQN network
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        feat = self.feature(x)
        val = self.value_stream(feat)
        adv = self.adv_stream(feat)
        return val + adv - adv.mean(1, keepdim=True)

# QR-DQN: Quantile Regression DQN network
class QuantileNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles=51, hidden=128):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        self.quantile_layer = nn.Linear(hidden, action_dim * num_quantiles)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.fc(x)
        quantiles = self.quantile_layer(x)
        quantiles = quantiles.view(batch_size, -1, self.num_quantiles)
        return quantiles  # shape: [batch, actions, quantiles]

# ============= Agent Base =============
class DQNAgent(ABC):
    @abstractmethod
    def select_action(self, state): pass
    @abstractmethod
    def learn(self): pass

# ============ DQN Variants ============
class VanillaDQN(DQNAgent):
    def __init__(self, state_dim, action_dim, args):
        self.device = args.device
        self.policy = QNetwork(state_dim, action_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.buffer = ReplayBuffer(args.buffer_size)
        self.steps = 0
        self.args = args
        self.action_dim = action_dim

    def select_action(self, state):
        eps = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
              np.exp(-1. * self.steps / self.args.eps_decay)
        self.steps += 1
        if random.random() < eps:
            return random.randrange(self.action_dim)
        state_v = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy(state_v).argmax().item()

    def learn(self):
        if len(self.buffer) < self.args.batch_size:
            return 0
        s, a, r, s2, d = self.buffer.sample(self.args.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        q_pred = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).max(1)[0]
            q_target = r + self.args.gamma * q_next * (1 - d)
        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.args.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return loss.item()

class DoubleDQN(VanillaDQN):
    def learn(self):
        if len(self.buffer) < self.args.batch_size:
            return 0
        s, a, r, s2, d = self.buffer.sample(self.args.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        q_pred = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_action = self.policy(s2).argmax(1)
            q_next = self.target(s2).gather(1, next_action.unsqueeze(1)).squeeze(1)
            q_target = r + self.args.gamma * q_next * (1 - d)
        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.args.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return loss.item()

class DuelingDQN(VanillaDQN):
    def __init__(self, state_dim, action_dim, args):
        super().__init__(state_dim, action_dim, args)
        self.policy = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)

class PER(DQNAgent):
    def __init__(self, state_dim, action_dim, args):
        self.device = args.device
        self.policy = QNetwork(state_dim, action_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.buffer = PrioritizedReplayBuffer(args.buffer_size)
        self.steps = 0
        self.args = args

    def select_action(self, state):
        eps = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
              np.exp(-1. * self.steps / self.args.eps_decay)
        self.steps += 1
        if random.random() < eps:
            return random.randrange(self.policy.layers[-1].out_features)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy(state).argmax().item()

    def learn(self):
        if len(self.buffer) < self.args.batch_size:
            return 0
        s, a, r, s2, d, idxs, ws = self.buffer.sample(self.args.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).to(self.device)
        ws = torch.FloatTensor(ws).to(self.device)

        q_pred = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_action = self.policy(s2).argmax(1)
            q_next = self.target(s2).gather(1, next_action.unsqueeze(1)).squeeze(1)
            q_target = r + self.args.gamma * q_next * (1 - d)
        loss = (ws * (q_pred - q_target).pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities
        priorities = (q_pred - q_target).abs().cpu().detach().numpy() + 1e-6
        self.buffer.update_priorities(idxs, priorities)

        if self.steps % self.args.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return loss.item()

class QRDQN(DQNAgent):
    def __init__(self, state_dim, action_dim, args):
        self.device = args.device
        self.num_quantiles = args.num_quantiles
        self.policy = QuantileNetwork(state_dim, action_dim, args.num_quantiles).to(self.device)
        self.target = QuantileNetwork(state_dim, action_dim, args.num_quantiles).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.buffer = ReplayBuffer(args.buffer_size)
        self.steps = 0
        self.args = args

    def select_action(self, state):
        eps = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
              np.exp(-1. * self.steps / self.args.eps_decay)
        self.steps += 1
        if random.random() < eps:
            return random.randrange(self.policy.quantile_layer.out_features // self.num_quantiles)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            quantiles = self.policy(state)  # shape [1, actions, quantiles]
            q_vals = quantiles.mean(2)
            return q_vals.argmax().item()

    def learn(self):
        if len(self.buffer) < self.args.batch_size:
            return 0
        s, a, r, s2, d = self.buffer.sample(self.args.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        # get quantiles
        quantiles = self.policy(s)  # [batch, actions, num_quantiles]
        batch_idx = torch.arange(self.args.batch_size, device=self.device)
        chosen_quantiles = quantiles[batch_idx, a, :]  # [batch, num_quantiles]

        with torch.no_grad():
            next_quantiles = self.target(s2)  # [batch, actions, num_quantiles]
            next_q_vals = next_quantiles.mean(2)
            next_actions = next_q_vals.argmax(1)
            next_chosen = next_quantiles[batch_idx, next_actions, :]
            target_quantiles = r.unsqueeze(1) + self.args.gamma * next_chosen * (1 - d.unsqueeze(1))

        # quantile regression loss
        tau = torch.linspace(0.0, 1.0, self.num_quantiles + 1)[1:]  # skip 0.0
        tau = tau.to(self.device)
        u = target_quantiles.unsqueeze(-1) - chosen_quantiles.unsqueeze(1)
        loss = (torch.abs(tau.unsqueeze(0) - (u.detach() < 0).float()) * u.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.args.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return loss.item()

# ============== Training & Testing ============
def train_and_evaluate(args):
    env = gym.make(args.env)
    env.reset(seed=args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # select agent type
    agent_map = {
        'dqn': VanillaDQN,
        'double': DoubleDQN,
        'dueling': DuelingDQN,
        'per': PER,
        'qrdqn': QRDQN
    }
    AgentClass = agent_map[args.method]
    agent = AgentClass(state_dim, action_dim, args)

    best_reward = -float('inf')
    best_model_path = None
    rewards = []
    losses = []

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        ep_loss = 0
        steps = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            # push transition
            if args.method == 'per':
                agent.buffer.push(state, action, reward, next_state, done)
            else:
                agent.buffer.push(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward
            loss = agent.learn()
            ep_loss += loss
            steps += 1
        avg_loss = ep_loss / steps if steps > 0 else 0
        rewards.append(ep_reward)
        losses.append(avg_loss)
        print(f"Episode: {ep:>4}/{args.episodes:<4} | Method: {args.method:<6} | Reward: {ep_reward:>8.2f} | Loss: {avg_loss:>9.4f}")

        # save best
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_model_path = f"best_{args.method}.pth"
            torch.save(agent.policy.state_dict(), best_model_path)

    env.close()

    # plot
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(losses); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(rewards); plt.title('Reward')
    plt.tight_layout()
    plt.savefig(f"{args.method}_training.png")
    plt.show()

    # test on best model
    print(f"Testing best model from {best_model_path}, reward threshold: {best_reward:.2f}")
    test_env = gym.make(args.env, render_mode='rgb_array')
    test_env = gym.wrappers.RecordVideo(test_env, args.video_dir, name_prefix=args.method)
    # test_agent = AgentClass(state_dim, action_dim, args)
    # test_agent.policy.load_state_dict(torch.load(best_model_path))
    # test_agent.policy.to(args.device)
    # test_agent.epsilon = 0.0
    test_agent = agent

    state, _ = test_env.reset(seed=args.seed)
    done = False
    
    test_agent.policy.eval()
    while not done:
        action = test_agent.select_action(state)
        state, _, done, _, _ = test_env.step(action)
    test_env.close()
    print(f"Video saved to {args.video_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLander-v3')
    parser.add_argument('--method', type=str, default='dqn', choices=['dqn','double','dueling','per','qrdqn'])
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.01)
    parser.add_argument('--eps_decay', type=int, default=50000)
    parser.add_argument('--target_update', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=111550159)
    parser.add_argument('--video_dir', type=str, default='videos')
    parser.add_argument('--num_quantiles', type=int, default=51)
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()

    os.makedirs(args.video_dir, exist_ok=True)
    train_and_evaluate(args)
