# ===== dqn.py =====
import random
import torch
import torch.nn as nn
import torch.optim as optim
from utils import DEVICE, ReplayBuffer

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(7*7*64, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_shape, num_actions, args):
        self.device = DEVICE
        self.num_actions = num_actions
        self.policy_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.memory = ReplayBuffer(args.memory_size)
        # Epsilon setup
        self.eps_start, self.eps_end, self.eps_decay = args.eps_start, args.eps_end, args.eps_decay
        self.epsilon = args.eps_start
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_decay)
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(st).argmax(dim=1).item()

    def optimize(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        st = torch.FloatTensor(states).to(self.device)
        acts = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rews = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_vals = self.policy_net(st).gather(1, acts)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1)[0].unsqueeze(1)
            target = rews + gamma * next_q * (1 - dones)
        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()