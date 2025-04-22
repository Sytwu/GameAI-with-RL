# ===== qr_dqn.py =====
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from utils import DEVICE, ReplayBuffer

class QRDQNAgent:
    def __init__(self, state_shape, num_actions, args):
        self.device = DEVICE
        self.num_actions = num_actions
        self.num_quantiles = args.num_quantiles
        self.quantile_tau = torch.linspace(0.0, 1.0, self.num_quantiles + 1)[1:-1].to(self.device)
        c,h,w = state_shape
        self.network = nn.Sequential(
            nn.Conv2d(c,32,8,4), nn.ReLU(),
            nn.Conv2d(32,64,4,2), nn.ReLU(),
            nn.Conv2d(64,64,3,1), nn.ReLU(),
            nn.Flatten(), nn.Linear(7*7*64,512), nn.ReLU(),
            nn.Linear(512, num_actions * self.num_quantiles)
        ).to(self.device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.lr)
        self.memory = ReplayBuffer(args.memory_size)

    def select_action(self, state):
        with torch.no_grad():
            quantiles = self.network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            quantiles = quantiles.view(-1, self.num_quantiles)
            q_vals = quantiles.mean(dim=1)
            return q_vals.argmax().item()

    def optimize(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        st = torch.FloatTensor(states).to(self.device)
        acts = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rews = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute quantile distribution
        quantiles = self.network(st).view(batch_size, self.num_quantiles, -1)
        with torch.no_grad():
            next_quantiles = self.target_network(ns).view(batch_size, self.num_quantiles, -1)
            next_q = next_quantiles.mean(dim=1)
            next_actions = next_q.argmax(dim=1)
            next_quantiles = next_quantiles[range(batch_size), :, next_actions]
            target_quantiles = rews + gamma * next_quantiles * (1 - dones)

        # Compute Huber loss for quantile regression
        quantile_tau = self.quantile_tau.view(1, -1, 1)
        u = target_quantiles.unsqueeze(1) - quantiles
        huber = torch.where(u.abs() <= 1.0, 0.5 * u.pow(2), u.abs() - 0.5)
        loss = (torch.abs(quantile_tau - (u.detach() < 0).float()) * huber).mean()

        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()