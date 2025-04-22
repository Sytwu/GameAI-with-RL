# ===== double_dqn.py =====
from dqn import DQNAgent
import torch.nn as nn
import torch

class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_shape, num_actions, args):
        super().__init__(state_shape, num_actions, args)

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
            next_actions = self.policy_net(ns).argmax(1, keepdim=True)
            next_q = self.target_net(ns).gather(1, next_actions)
            target = rews + gamma * next_q * (1 - dones)
        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()