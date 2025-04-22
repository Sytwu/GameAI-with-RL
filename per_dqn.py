# ===== per_dqn.py =====
import torch.nn as nn
import torch.optim as optim
import torch
from utils import DEVICE, PrioritizedReplayBuffer
from dqn import DQN, DQNAgent

class PERAgent(DQNAgent):
    def __init__(self, state_shape, num_actions, args):
        self.device = DEVICE
        self.num_actions = num_actions
        self.policy_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha)
        self.eps_start, self.eps_end, self.eps_decay = args.eps_start, args.eps_end, args.eps_decay
        self.epsilon = args.eps_start
        self.steps_done = 0
        self.beta = args.per_beta

    def optimize(self, batch_size, gamma):
        if len(self.memory.buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(batch_size, self.beta)
        st = torch.FloatTensor(states).to(self.device)
        acts = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rews = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        q_vals = self.policy_net(st).gather(1, acts)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1)[0].unsqueeze(1)
            target = rews + gamma * next_q * (1 - dones)
        loss = (q_vals - target).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.memory.update_priorities(indices, prios.data.cpu().numpy())