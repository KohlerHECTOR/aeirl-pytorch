import os
import json
import pickle
import argparse

import torch
import gym

import shutil

from stable_baselines3 import PPO
from models.nets import Expert
from models.gail import GAIL


def main(env_name, envs_mujoco, path_save_log="default_save"):
    if path_save_log == "default_save":
        if os.path.isdir(path_save_log):
            try:
                shutil.rmtree(path_save_log)
            except OSError as e:
                print(e)
            else:
                print("The existing file: "+path_save_log+" is clear")
        else:
            os.mkdir(path_save_log)
            os.mkdir(path_save_log+'/log')
            path_save_log += '/log'

    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    if env_name not in ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3"]+envs_mujoco:
        print("The environment name is wrong!")
        return

    expert_ckpt_path = "experts"
    expert_ckpt_path = os.path.join(expert_ckpt_path, env_name)

    ckpt_path = os.path.join(ckpt_path, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[env_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = gym.make(env_name)
    env.reset()

    state_dim = len(env.observation_space.high)
    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if env_name in envs_mujoco:
        expert = PPO.load(os.path.join(
            expert_ckpt_path, f"PPO-{env_name}"))

    else:
        with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
            expert_config = json.load(f)
        expert = Expert(state_dim, action_dim, discrete,
                        **expert_config).to(device)
        expert.pi.load_state_dict(torch.load(os.path.join(
            expert_ckpt_path, "policy.ckpt"), map_location=device))

    model = GAIL(state_dim, action_dim, discrete, config,
                 path_save_log=path_save_log).to(device)

    results = model.train(env, expert, envs_mujoco)

    env.close()

    with open(os.path.join(ckpt_path, "gail_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(
            model.pi.state_dict(), os.path.join(ckpt_path, "gail_policy.ckpt")
        )
    if hasattr(model, "v"):
        torch.save(
            model.v.state_dict(), os.path.join(ckpt_path, "gail_value.ckpt")
        )
    if hasattr(model, "d"):
        torch.save(
            model.d.state_dict(), os.path.join(ckpt_path, "gail_discriminator.ckpt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="BipedalWalker-v3",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    parser.add_argument('--envs_mujoco', nargs='+',
                        default=["Hopper-v2", "Swimmer-v2"])
    args = parser.parse_args()

    main(**vars(args))
