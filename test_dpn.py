# test_dqn.py
# 用於測試已訓練的 DQN 模型並將遊戲場景 render 出來

import numpy as np
import torch
import cv2
import gymnasium as gym
import ale_py
import argparse

from collections import deque

gym.register_envs(ale_py)

# 如果有單獨定義的模型模塊，可直接 import，否則在此重新定義 DQN 結構
# 這裡假設 DQN 定義在 dqn_model.py 中：

import torch.nn as nn
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    def forward(self, x):
        return self.net(x)


# 前處理函數：同訓練時

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = np.array(resized, dtype=np.float32) / 255.0
    return normalized


def test(env_name, model_path, num_episodes=5, seed=42):
    # 建立環境並啟用 render
    env = gym.make(env_name, render_mode="human")
    env.reset(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {device}")

    # 初始化模型
    env_reset = env.reset()[0]
    sample_frame = preprocess_frame(env_reset)
    state_shape = (4, 84, 84)
    num_actions = env.action_space.n

    policy_net = DQN(state_shape, num_actions).to(device)
    # 載入已訓練的參數
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    for episode in range(1, num_episodes+1):
        obs, _ = env.reset()
        frame = preprocess_frame(obs)
        state_deque = deque([frame] * 4, maxlen=4)
        state = np.stack(state_deque, axis=0)
        done = False
        total_reward = 0.0

        while not done:
            # 選擇動作 (greedy)
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()

            # 進行一步
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # 更新狀態
            next_frame = preprocess_frame(next_obs)
            state_deque.append(next_frame)
            state = np.stack(state_deque, axis=0)

        print(f"Episode {episode}: Total Reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained DQN on Atari environment")
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5", help="Gym environment name")
    parser.add_argument("--model", type=str, default="best_dqn_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    args = parser.parse_args()

    test(env_name=args.env, model_path=args.model, num_episodes=args.episodes)
