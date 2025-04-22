# ===== dueling_dqn.py =====
import torch.nn as nn
import torch
import torch.optim as optim
from utils import DEVICE, ReplayBuffer
from double_dqn import DoubleDQNAgent

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        c,_,_ = input_shape
        self.feature = nn.Sequential(
            nn.Conv2d(c,32,8,4), nn.ReLU(),
            nn.Conv2d(32,64,4,2), nn.ReLU(),
            nn.Conv2d(64,64,3,1), nn.ReLU(),
            nn.Flatten()
        )
        self.value = nn.Sequential(nn.Linear(7*7*64,512), nn.ReLU(), nn.Linear(512,1))
        self.adv = nn.Sequential(nn.Linear(7*7*64,512), nn.ReLU(), nn.Linear(512,num_actions))

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.adv(f)
        return v + (a - a.mean(dim=1, keepdim=True))

class DuelingAgent(DoubleDQNAgent):
    def __init__(self, state_shape, num_actions, args):
        self.device = DEVICE
        self.num_actions = num_actions
        self.policy_net = DuelingDQN(state_shape, num_actions).to(self.device)
        self.target_net = DuelingDQN(state_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.memory = ReplayBuffer(args.memory_size)
        # inherit eps parameters
        self.epsilon, self.eps_start, self.eps_end, self.eps_decay = args.eps_start, args.eps_start, args.eps_end, args.eps_decay
        self.steps_done = 0