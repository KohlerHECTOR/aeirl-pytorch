
import argparse
from ast import boolop
import scipy.interpolate as intplt

from sqlalchemy import true

from models.nets import AE, Discriminator
from stable_baselines3 import PPO
import numpy as np
import gym
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import torch
import tqdm
from torch import FloatTensor
import re


def tokenize(filename):
    digits = re.compile(r'(\d+)')
    return tuple(int(token) if match else token
                 for token, match in
                 ((fragment, digits.search(fragment))
                  for fragment in digits.split(filename)))


def to_plot(data, label=None, color='blue', smooth=True):
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    if smooth:
        x_new = np.linspace(0, len(mean_data), len(mean_data)*4)
        a_BSpline = intplt.make_interp_spline(
            np.arange(len(mean_data)), mean_data)
        y_new = a_BSpline(x_new)
        plt.plot(x_new, y_new, label=label, color=color)
    else:
        plt.plot(mean_data, label=label, color=color)
    plt.fill_between(np.arange(len(mean_data)), mean_data +
                     std_data, mean_data - std_data, alpha=0.1)


def main(noisy):

    env = gym.make(args.env_name)
    state_dim = len(env.observation_space.high)
    action_dim = env.action_space.shape[0]

    reward_AE = AE(state_dim, action_dim, False)
    reward_Discr = Discriminator(state_dim, action_dim, False)

    if noisy:
        reward_AE.load_state_dict(torch.load(
            f"final_reward_nets/{args.env_name}/aeirl_autoencoder_0.3_.ckpt", map_location='cpu'))
        reward_Discr.load_state_dict(torch.load(
            f"final_reward_nets/{args.env_name}/gail_discriminator_0.3_.ckpt", map_location='cpu'))
    else:
        reward_AE.load_state_dict(torch.load(
            f"final_reward_nets/{args.env_name}/aeirl_autoencoder_0_.ckpt", map_location='cpu'))

        reward_Discr.load_state_dict(torch.load(
            f"final_reward_nets/{args.env_name}/gail_discriminator_0_.ckpt", map_location='cpu'))

    ppo_difs_levels_path = [f for f in listdir(
        args.checkpoint_path+args.env_name) if isfile(join(args.checkpoint_path+args.env_name, f))]
    ppo_difs_levels_path.sort(key=tokenize)

    all_exps_AE = []
    all_exps_Discr = []
    for exp in range(args.nb_exp):
        exp_results_AE = []
        exp_results_Discr = []
        traj_length = []
        env.seed(exp)

        boolean = True

        for ppo_policy in tqdm.tqdm(ppo_difs_levels_path[:-1]):
            policy = PPO.load(
                args.checkpoint_path+args.env_name+"/"+ppo_policy)
            state = env.reset()
            sum_rewards_AE = 0
            sum_rewards_Discr = 0
            steps_done = 0
            for _ in range(args.steps_per_traj):

                action = policy.predict(state, deterministic=True)[0]
                next_state, _, done, _ = env.step(action)
                steps_done += 1
                state = FloatTensor(np.array([state]))
                action = FloatTensor(np.array([action]))

                sum_rewards_AE += 1/(1+reward_AE(state, action))
                sum_rewards_Discr += -torch.log(reward_Discr(state, action))

                if done:
                    break
                state = next_state

            if boolean:
                boolean = False
                random_rew = sum_rewards_AE.item()/steps_done, sum_rewards_Discr.item()/steps_done
            else:
                traj_length.append(steps_done)
                exp_results_AE.append(sum_rewards_AE.item())
                exp_results_Discr.append(sum_rewards_Discr.item())
            expert_rew = sum_rewards_AE.item()/steps_done, sum_rewards_Discr.item()/steps_done

        for i in range(len(exp_results_AE)):

            exp_results_AE[i] = (
                exp_results_AE[i]/traj_length[i])
            exp_results_Discr[i] = (
                exp_results_Discr[i] / traj_length[i])

            exp_results_AE[i] = (exp_results_AE[i]-random_rew[0]
                                 )/(expert_rew[0]-random_rew[0])
            exp_results_Discr[i] = (
                exp_results_Discr[i]-random_rew[1])/(expert_rew[1]-random_rew[1])

        all_exps_AE.append(exp_results_AE)
        all_exps_Discr.append(exp_results_Discr)

    if noisy:
        to_plot(all_exps_AE, label="AEIRL noisy", color="violet")
        to_plot(all_exps_Discr, label="GAIL noisy", color="green")
    else:
        to_plot(all_exps_AE, label="AEIRL", color="blue")
        to_plot(all_exps_Discr, label="GAIL", color="red")
    plt.legend()
    plt.xlabel("Level")
    plt.ylabel("learned rewards")
    plt.title(f"Trajectory Space - {args.env_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="Hopper-v2",
        help="Type the environment name to run"
    )
    parser.add_argument(
        "--nb_exp",
        type=int,
        default=10,
        help="Number of run"
    )
    parser.add_argument(
        "--steps_per_traj",
        type=int,
        default=5000,
        help="Number of step for each traj"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="experts/create-expert/checkpoints/",
        help="path of the checkpoints of the trained agent"
    )
    args = parser.parse_args()

    main(noisy=True)
    main(noisy=False)
    plt.show()
