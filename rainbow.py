import copy
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from utils import DEVICE, PrioritizedReplayBuffer, NoisyLinear


def build_feature_extractor(in_channels: int) -> nn.Sequential:
    """
    Build the convolutional feature extractor.

    Args:
        in_channels (int): Number of input channels.
    Returns:
        nn.Sequential: Convolutional network for feature extraction.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )


def projection_distribution(
    next_dist: torch.Tensor,
    support: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    vmin: float,
    vmax: float,
    num_atoms: int,
    gamma: float,
    n_step: int,
) -> torch.Tensor:
    """
    Compute the projected distribution for multi-step returns.

    Args:
        next_dist (torch.Tensor): Next state distribution [batch, atoms].
        support (torch.Tensor): Atom support values.
        rewards (torch.Tensor): Rewards for the transitions.
        dones (torch.Tensor): Done flags for the transitions.
        vmin (float): Minimum return value.
        vmax (float): Maximum return value.
        num_atoms (int): Number of atoms.
        gamma (float): Discount factor.
        n_step (int): Number of steps for multi-step returns.
    Returns:
        torch.Tensor: Projected distribution [batch, atoms].
    """
    delta = (vmax - vmin) / (num_atoms - 1)
    # Compute projected atom values
    tz = rewards.unsqueeze(1) + (gamma ** n_step) * support.unsqueeze(0) * (1 - dones.unsqueeze(1))
    tz = tz.clamp(vmin, vmax)
    b = (tz - vmin) / delta
    l = b.floor().long()
    u = b.ceil().long()

    m = torch.zeros_like(next_dist)
    for i in range(b.size(0)):
        for j in range(num_atoms):
            m[i, l[i, j]] += next_dist[i, j] * (u[i, j] - b[i, j])
            m[i, u[i, j]] += next_dist[i, j] * (b[i, j] - l[i, j])
    return m


class RainbowNetwork(nn.Module):
    """
    Network architecture for Rainbow DQN, including value and advantage streams.
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        num_actions: int,
        num_atoms: int,
    ) -> None:
        super().__init__()
        c, h, w = in_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.feature_extractor = build_feature_extractor(c)

        # Calculate the flattened feature size after conv layers
        flattened_size = 64 * (((((h - 8) // 4 + 1) - 4) // 2 + 1) - 3 + 1) * (
            (((w - 8) // 4 + 1) - 4) // 2 + 1 - 3 + 1
        )
        self.value_head = nn.Sequential(
            NoisyLinear(flattened_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_atoms),
        )
        self.advantage_head = nn.Sequential(
            NoisyLinear(flattened_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * num_atoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        v = self.value_head(features).view(-1, 1, self.num_atoms)
        a = self.advantage_head(features).view(-1, self.num_actions, self.num_atoms)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q


class RainbowAgent:
    """
    Agent implementation for Rainbow DQN, managing networks and training logic.
    """

    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        num_actions: int,
        args,
    ) -> None:
        # Environment and hyperparameters
        self.device = DEVICE
        self.num_actions = num_actions
        self.num_atoms = args.num_atoms
        self.vmin = args.v_min
        self.vmax = args.v_max
        self.support = torch.linspace(self.vmin, self.vmax, self.num_atoms).to(self.device)

        # Networks and optimizer
        self.policy_net = RainbowNetwork(state_shape, num_actions, self.num_atoms).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)

        # Replay buffer and training settings
        self.memory = PrioritizedReplayBuffer(args.memory_size, args.per_alpha)
        self.gamma = args.gamma
        self.n_step = args.n_step
        self.eps = args.eps_start
        self.beta = args.per_beta

    def select_action(self, state: torch.Tensor) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            state (torch.Tensor): Current state tensor.
        Returns:
            int: Selected action index.
        """
        if random.random() < self.eps:
            return random.randrange(self.num_actions)
        state = state.unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.policy_net(state)
            q = (dist * self.support).sum(dim=-1)
        return q.argmax().item()

    def train_step(self, batch_size: int) -> None:
        """
        Perform a single training step from replay memory.

        Args:
            batch_size (int): Number of samples per batch.
        """
        if len(self.memory) < batch_size:
            return

        (states, actions, rewards, next_states, dones), weights, indices = \
            self.memory.sample(batch_size, self.beta)

        # Convert to tensors
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)
        weights = torch.tensor(weights, device=self.device)

        # Current distribution
        dist = self.policy_net(states).permute(0, 2, 1)
        curr_dist = dist[range(batch_size), :, actions]

        # Target distribution
        with torch.no_grad():
            next_dist = self.target_net(next_states).permute(0, 2, 1)
            next_q = (next_dist * self.support).sum(dim=1)
            next_actions = next_q.argmax(dim=1)
            next_dist = next_dist[range(batch_size), :, next_actions]
            target_dist = projection_distribution(
                next_dist, self.support, rewards, dones,
                self.vmin, self.vmax, self.num_atoms, self.gamma, self.n_step
            )

        # Compute loss and update priorities
        loss = - (target_dist * curr_dist.log()).sum(dim=1)
        priorities = loss.detach().cpu().numpy()
        loss = (loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priorities(indices, priorities)
