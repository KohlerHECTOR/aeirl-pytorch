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
    checkpoint_callback = CheckpointCallback(save_freq=max((args.total_timesteps/50)//args.n_envs, 1), save_path=f'./checkpoints/{args.env_name}/',
                                             name_prefix=str(args.env_name))

    env = make_vec_env(args.env_name, n_envs=args.n_envs)

    log_dir = "data/"
    os.makedirs(log_dir, exist_ok=True)

    with open(args.hyperparams_file) as json_file:
        hyperparams = json.load(json_file)
    hyperparams = hyperparams[args.env_name]
    if 'activation_fn' in hyperparams["policy_kwargs"]:
        if hyperparams["policy_kwargs"]["activation_fn"] == "nn.ReLU":
            hyperparams["policy_kwargs"]["activation_fn"] = torch.nn.ReLU

    model = PPO('MlpPolicy', env, tensorboard_log=log_dir,
                verbose=1, **hyperparams)

    model.learn(total_timesteps=args.total_timesteps,
                callback=checkpoint_callback)

    print(model.policy)
    model.save(args.save_file)

    time_one_env = time.time() - start_time_one_env
    print(f"learning took {time_one_env:.2f}s")

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=20, deterministic=True, render=args.render)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train a PPO agent.')
    parser.add_argument('--env_name', type=str, default="Walker2d-v2")
    parser.add_argument('--total_timesteps', type=int, default=int(4e6))
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--n_envs', type=int, default=16)
    parser.add_argument('--hyperparams_file', type=str,
                        default="hyperparams.json")
    parser.add_argument('--save_file', type=str,
                        default="../Walker2d-v2/PPO-Walker2d-v2-new")

    args = parser.parse_args()
    main()
