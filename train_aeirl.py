import os
import json
import pickle
import argparse

import torch
import gym

import shutil
from stable_baselines3 import PPO
from models.nets import Expert
from models.aeirl import AEIRL


def main(env_name, path_save_log="default_save", simu_nb = None, noise = 0):
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

    if env_name not in ["Hopper-v2", "Swimmer-v2", "Walker2d-v2"]: # ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3", "Hopper-v2", "Swimmer-v2", "Walker2d-v2"]:
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
    env.seed(simu_nb)
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

    if env_name in ["Hopper-v2", "Swimmer-v2", "Walker2d-v2"]:
        expert = PPO.load(os.path.join(
            expert_ckpt_path, f"PPO-{env_name}"))

    else:
        with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
            expert_config = json.load(f)
        expert = Expert(state_dim, action_dim, discrete,
                        **expert_config).to(device)
        expert.pi.load_state_dict(
            torch.load(
                os.path.join(expert_ckpt_path, "policy.ckpt"), map_location=device
            )
        )

    model = AEIRL(state_dim, action_dim, discrete, config,
                  path_save_log=path_save_log).to(device)

    results = model.train(env, expert, noise = noise)

    env.close()

    # with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
    #     pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(
            model.pi.state_dict(), os.path.join(ckpt_path, "aeirl_policy_"+ str(noise) + "_.ckpt")
        )
    if hasattr(model, "v"):
        torch.save(
            model.v.state_dict(), os.path.join(ckpt_path, "aeirl_value_"+ str(noise) + "_.ckpt")
        )
    if hasattr(model, "d"):
        torch.save(
            model.d.state_dict(), os.path.join(ckpt_path, "aeirl_autoencoder_"+ str(noise) + "_.ckpt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="Hopper-v2",
        help="Type the environment name to run. \
            The possible environments are \
                [Hopper-v2, Swimmer-v2, Walker2d-v2]" # "[CartPole-v1, Pendulum-v0, BipedalWalker-v3, Hopper-v2, Swimmer-v2, Walker2d-v2]"
    )
    args = parser.parse_args()

    main(**vars(args))
