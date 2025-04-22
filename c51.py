# ===== c51.py =====
import torch.nn as nn
import torch.optim as optim
import torch
from utils import DEVICE, ReplayBuffer

class C51Agent:
    def __init__(self, state_shape, num_actions, args):
        self.device = DEVICE
        self.num_actions = num_actions
        self.v_min, self.v_max = args.v_min, args.v_max
        self.num_atoms = args.num_atoms
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.z = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.policy_net = self.build_model(state_shape, num_actions)
        self.target_net = self.build_model(state_shape, num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.memory = ReplayBuffer(args.memory_size)

    def build_model(self, input_shape, num_actions):
        c, h, w = input_shape
        return nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(7*7*64, 512), nn.ReLU(),
            nn.Linear(512, num_actions * self.num_atoms)
        ).to(self.device)

    def select_action(self, state):
        with torch.no_grad():
            logits = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            probs = logits.view(-1, self.num_atoms).softmax(dim=1)
            q_vals = (probs * self.z).sum(dim=1)
            return q_vals.argmax().item()

    def optimize(self, batch_size, gamma):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        st = torch.FloatTensor(states).to(self.device)
        acts = torch.LongTensor(actions).to(self.device)
        rews = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Distribution projection
        batch_size = len(states)
        with torch.no_grad():
            next_logits = self.target_net(ns)
            next_probs = (next_logits.view(-1, self.num_actions, self.num_atoms).softmax(dim=2))
            next_q = (next_probs * self.z).sum(dim=2)
            next_actions = next_q.argmax(1)
            next_dist = next_probs[range(batch_size), next_actions]
            t_z = rews.unsqueeze(1) + gamma * self.z.unsqueeze(0) * (1 - dones.unsqueeze(1))
            t_z = t_z.clamp(self.v_min, self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            m = torch.zeros(batch_size, self.num_atoms).to(self.device)
            for i in range(batch_size):
                for j in range(self.num_atoms):
                    m[i, l[i,j]] += next_dist[i,j] * (u[i,j] - b[i,j])
                    m[i, u[i,j]] += next_dist[i,j] * (b[i,j] - l[i,j])

        logits = self.policy_net(st)
        log_probs = logits.view(-1, self.num_actions, self.num_atoms).log_softmax(dim=2)
        loss = -(m * log_probs[range(batch_size), acts]).sum(1).mean()

        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()