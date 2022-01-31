import gym
import torch.utils.tensorboard
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import time
import json
import torch
import gym


def main():
    start_time_one_env = time.time()
    checkpoint_callback = CheckpointCallback(save_freq=args.total_timesteps/50, save_path=f'./policies/{args.env_name}/',
                                             name_prefix=str(args.env_name))

    env = make_vec_env(args.env_name, n_envs=args.n_envs)

    log_dir = "data/"
    os.makedirs(log_dir, exist_ok=True)
    file_name = args.file_name
    log_file_name = log_dir + file_name

    with open(args.hyperparams_file) as json_file:
        hyperparams = json.load(json_file)
    hyperparams = hyperparams[args.env_name]
    if "activation_fn" in hyperparams:
        hyperparams["policy_kwargs"]["activation_fn"] = torch.nn.ReLU

    model = PPO('MlpPolicy', env, tensorboard_log=log_file_name,
                verbose=1, **hyperparams)

    model.learn(total_timesteps=args.total_timesteps,
                callback=checkpoint_callback)

    print(model.policy)
    model.save(args.save_file)

    time_one_env = time.time() - start_time_one_env
    print(f"learning took {time_one_env:.2f}s")

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=args.n_eval_episodes, deterministic=True, render=args.render)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Launch pybullet simulation run.')
    parser.add_argument('--env_name', type=str, default="Walker2d-v2")
    parser.add_argument('--total_timesteps', type=int, default=int(1e6))
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--file_name', type=str, default="")
    parser.add_argument('--n_eval_episodes', type=int, default=20)
    parser.add_argument('--hyperparams_file', type=str,
                        default="hyperparams.json")
    parser.add_argument('--save_file', type=str,
                        default="Walker2d_model")

    args = parser.parse_args()
    main()
