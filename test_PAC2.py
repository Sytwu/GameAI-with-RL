import argparse
import os
import csv
import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO

# Register ALE environments
gym.register_envs(ale_py)

# Death penalty wrapper (re-used from training script)
class DeathPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, death_penalty=1.0):
        super().__init__(env)
        self.death_penalty = death_penalty
        self.lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get('lives', 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        curr_lives = info.get('lives', self.lives)
        if curr_lives < self.lives:
            # Apply death penalty and force auto-action
            reward -= self.death_penalty
            obs2, extra_r, term2, trunc2, info2 = self.env.step(1)
            reward += extra_r
            obs = obs2
            info.update(info2)
            curr_lives = info2.get('lives', curr_lives)
        self.lives = curr_lives
        return obs, reward, terminated, truncated, info


def get_model_class(algo_name):
    if algo_name == 'a2c':
        return A2C
    if algo_name == 'ppo':
        return PPO
    if algo_name == 'trpo':
        return TRPO
    raise ValueError(f"Unsupported algorithm: {algo_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Load a trained RL agent, run evaluation episodes, and record video"
    )
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the saved model (without extension)')
    parser.add_argument('--algo', choices=['a2c', 'ppo', 'trpo'], required=True,
                        help='Which algorithm was used (a2c, ppo, trpo)')
    parser.add_argument('--env', default='ALE/Breakout-v5',
                        help='Gym environment ID')
    parser.add_argument('--death-penalty', type=float, default=1.0,
                        help='Death penalty used in wrapper')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of evaluation episodes')
    parser.add_argument('--csv-file', type=str, default='eval_rewards.csv',
                        help='CSV to write episode rewards')
    parser.add_argument('--video-folder', type=str, default='videos',
                        help='Directory to save recorded videos')
    args = parser.parse_args()

    # Prepare CSV output
    os.makedirs(os.path.dirname(args.csv_file) or '.', exist_ok=True)
    with open(args.csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward'])

    # Make sure video folder exists
    video_folder = os.path.join(args.video_folder, args.algo)
    os.makedirs(video_folder, exist_ok=True)

    # Build environment with rgb_array and video recording
    env = gym.make(args.env, render_mode='rgb_array')
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: True,   # Record every episode
        name_prefix=f"{os.path.basename(args.model_path)}_ep"
    )
    env = DeathPenaltyWrapper(env, death_penalty=args.death_penalty)

    # Load trained model
    ModelClass = get_model_class(args.algo)
    model = ModelClass.load(args.model_path)

    # Evaluation loop
    all_rewards = []
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        lives = 5

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            cur_lives = info.get('lives')
            if lives != cur_lives:
                obs2, extra_r, term2, trunc2, info2 = env.step(1)
                total_reward += extra_r
                obs = obs2
                info.update(info2)
                cur_lives = info2.get('lives')
            lives = cur_lives

        print(f"Episode {ep}: Reward = {total_reward:.2f}")
        all_rewards.append(total_reward)

        # Append to CSV
        with open(args.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, total_reward])

    # Summary
    avg_reward = sum(all_rewards) / len(all_rewards)
    print(f"Average reward over {args.episodes} episodes: {avg_reward:.2f}")
    print(f"Videos saved to: {os.path.abspath(video_folder)}")

    env.close()


if __name__ == '__main__':
    main()
