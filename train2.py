import os
import argparse
import gymnasium as gym
import ale_py
import numpy as np
import torch
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import preprocess_frame, get_logger
from dqn import DQNAgent
from double_dqn import DoubleDQNAgent
from dueling_dqn import DuelingAgent
from per_dqn import PERAgent
from c51 import C51Agent
from qr_dqn import QRDQNAgent
from rainbow import RainbowAgent

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gym.register_envs(ale_py)
line = '-'*50

def make_agent(
    algo_name: str,
    state_shape: Tuple[int, int, int],
    num_actions: int,
    args: argparse.Namespace
) -> object:
    """
    Factory function to create the specified agent.

    Args:
        algo_name (str): Algorithm key (e.g., 'dqn', 'rainbow').
        state_shape (tuple): Shape of input state (channels, height, width).
        num_actions (int): Number of discrete actions.
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        Agent instance.
    """
    agent_map = {
        'dqn': DQNAgent,
        'double': DoubleDQNAgent,
        'dueling': DuelingAgent,
        'per': PERAgent,
        'c51': C51Agent,
        'qr': QRDQNAgent,
        'rainbow': RainbowAgent,
    }
    try:
        AgentClass = agent_map[algo_name]
    except KeyError:
        raise ValueError(f"Algorithm '{algo_name}' is not supported.")
    return AgentClass(state_shape, num_actions, args)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training.
    """
    parser = argparse.ArgumentParser(description="Train various DQN variants on Atari environments.")
    parser.add_argument('--algo', choices=list(
        ['dqn', 'double', 'dueling', 'per', 'c51', 'qr', 'rainbow']
    ), required=True, help='Which algorithm to train.')
    parser.add_argument('--env', default='ALE/Breakout-v5', help='Gym environment id.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--episodes', type=int, default=int(1e5), help='Total number of episodes.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for updates.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--memory_size', type=int, default=100000, help='Replay buffer capacity.')
    parser.add_argument('--eps_start', type=float, default=1.0, help='Starting epsilon value.')
    parser.add_argument('--eps_end', type=float, default=0.1, help='Final epsilon value.')
    parser.add_argument('--eps_decay', type=float, default=1e-6, help='Epsilon decay rate per step.')
    parser.add_argument('--target_update', type=int, default=1000, help='Target network update frequency (steps).')
    parser.add_argument('--death_penalty', type=float, default=1.0, help='Penalty for losing a life.')
    # Prioritized Experience Replay args
    parser.add_argument('--per_alpha', type=float, default=0.6, help='PER alpha parameter.')
    parser.add_argument('--per_beta', type=float, default=0.4, help='PER beta parameter.')
    # C51 args
    parser.add_argument('--v_min', type=float, default=-10, help='Minimum support value for C51.')
    parser.add_argument('--v_max', type=float, default=10, help='Maximum support value for C51.')
    parser.add_argument('--num_atoms', type=int, default=51, help='Number of atoms for C51.')
    # QR-DQN args
    parser.add_argument('--num_quantiles', type=int, default=51, help='Number of quantiles for QR-DQN.')

    return parser.parse_args()


def main() -> None:
    """
    Main training loop.
    """
    args = parse_args()
    logger = get_logger(log_file=f"training_{args.algo}.txt")

    # Initialize environment
    env = gym.make(args.env)
    env.reset(seed=args.seed)
    obs, info = env.reset()

    # Preprocess initial state
    frame = preprocess_frame(obs)
    state_shape = (4, 84, 84)  # Stacked frame shape
    num_actions = env.action_space.n
    agent = make_agent(args.algo, state_shape, num_actions, args)

    total_steps = 0
    best_reward = -float('inf')
    recent_rewards: List[float] = []

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset()
        lives = info.get('lives', 0)
        state_stack = [preprocess_frame(obs)] * 4
        episode_reward = 0.0
        done = False

        while not done:
            state = np.stack(state_stack, axis=0)
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Penalize life loss
            current_lives = info.get('lives', lives)
            if current_lives < lives:
                reward -= args.death_penalty
                lives = current_lives

            # Update state stack
            next_frame = preprocess_frame(next_obs)
            state_stack.append(next_frame)
            state_stack.pop(0)

            next_state = np.stack(state_stack, axis=0)
            episode_reward += reward
            total_steps += 1

            # Store transition
            if hasattr(agent, 'memory'):
                agent.memory.add(state, action, reward, next_state, done)

            # Perform learning step
            agent.train_step(args.batch_size)

            # Update target network periodically
            if total_steps % args.target_update == 0 and hasattr(agent, 'target_net'):
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

        # Logging and saving
        recent_rewards.append(episode_reward)
        avg_last_100 = np.mean(recent_rewards[-100:])
        logger.info(
            f"Episode {episode} | Reward: {episode_reward:.2f} | "
            f"Avg100: {avg_last_100:.2f} | Epsilon: {getattr(agent, 'eps', 0):.3f}"
        )
        print(
            f"Ep {episode} | R: {episode_reward:.2f} | "
            f"Avg100: {avg_last_100:.2f} | Eps: {getattr(agent, 'eps', 0):.3f}"
        )

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.policy_net.state_dict(), f"best_{args.algo}.pth")
            logger.info(f"{line}> New best {best_reward:.2f}, model saved.")

        # Periodic report
        if episode % 100 == 0:
            print(f"Completed {episode} episodes. Last 100-episode avg: {avg_last_100:.2f}")

if __name__ == '__main__':
    main()
