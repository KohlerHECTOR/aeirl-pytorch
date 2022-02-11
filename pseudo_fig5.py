# import AE
# import Discriminator
# import PolicyNet
import argparse

from models.nets import AE, Discriminator
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import gym
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import torch
import tqdm
if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor
import re


def tokenize(filename):
    digits = re.compile(r'(\d+)')
    return tuple(int(token) if match else token
                 for token, match in
                 ((fragment, digits.search(fragment))
                  for fragment in digits.split(filename)))


def to_plot(data, label=None):
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    plt.plot(mean_data, label=label)
    plt.fill_between(np.arange(len(mean_data)), mean_data +
                     std_data, mean_data - std_data, alpha=0.1)


def main():
    env = gym.make(args.env_name)
    state_dim = len(env.observation_space.high)
    action_dim = env.action_space.shape[0]

    reward_AE = AE(state_dim, action_dim, False)
    reward_AE.load_state_dict(torch.load(
        f"ckpts/{args.env_name}/aeirl_autoencoder.ckpt"))

    reward_Discr = Discriminator(state_dim, action_dim, False)
    reward_Discr.load_state_dict(torch.load(
        f"ckpts/{args.env_name}/gail_discriminator.ckpt"))

    ppo_difs_levels_path = [f for f in listdir(
        f"create-expert/policies/{args.env_name}") if isfile(join(f"create-expert/policies/{args.env_name}", f))]
    ppo_difs_levels_path.sort(key=tokenize)

    # print(ppo_difs_levels_path, len(ppo_difs_levels_path))

    all_exps_AE = []
    all_exps_Discr = []
    for exp in range(args.nb_exp):
        exp_results_AE = []
        exp_results_Discr = []
        exp_results_expert = []
        traj_length = []
        env.seed(exp)
        for ppo_policy in tqdm.tqdm(ppo_difs_levels_path):
            # print(ppo_policy)
            policy = PPO.load(
                f"create-expert/policies/{args.env_name}/"+ppo_policy)
            state = env.reset()
            sum_rewards_AE = 0
            sum_rewards_Discr = 0
            sum_rewards_expert = 0

            # mean_reward, std_reward = evaluate_policy(
            #     policy, env, n_eval_episodes=5, deterministic=True)
            # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

            # evaluate intermediate ppo policy using AE or Discr reward
            steps_done = 0
            for steps in range(args.steps_per_traj):

                action = policy.predict(state)[0]
                next_state, reward, done, info = env.step(action)
                steps_done += 1
                state = FloatTensor(np.array([state]))
                action = FloatTensor(np.array([action]))
                # print(state, action)
                sum_rewards_expert += reward
                sum_rewards_AE += 1 / \
                    (1 + reward_AE(state, action))
                sum_rewards_Discr += (-1)*torch.log(
                    reward_Discr(state, action))
                # sum_rewards_Discr += (reward_Discr(state, action))

                if done:
                    break
                state = next_state

            print("sum", sum_rewards_AE.item(),
                  sum_rewards_Discr.item(), sum_rewards_expert)

            traj_length.append(steps_done)
            exp_results_AE.append(sum_rewards_AE.item())
            exp_results_Discr.append(sum_rewards_Discr.item())
            exp_results_expert.append(sum_rewards_expert.item())

            print()

        for i in range(len(exp_results_AE)):
            # print(exp_results_AE[i], exp_results_AE[i]/traj_length[i])
            exp_results_AE[i] = (
                exp_results_AE[i]/traj_length[i])
            exp_results_Discr[i] = (
                exp_results_Discr[i] / traj_length[i])
            # exp_results_Discr[i] = scaled_normalized(
            #     exp_results_Discr[i], traj_length)
            # print(exp_results_AE[-1], exp_results_Discr)

        # print(exp_results_AE)
        all_exps_AE.append(exp_results_AE)
        all_exps_Discr.append(exp_results_Discr)

    to_plot(all_exps_AE, label="AEIRL")
    to_plot(all_exps_Discr, label="GAIL")
    plt.legend()
    plt.xlabel("Level")
    plt.ylabel("learned rewards")
    plt.title("Trajectory Space")
    plt.show()


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
        default=1,
        help="Number of run time"
    )
    parser.add_argument("--steps_per_traj", type=int, default=5000, help="")

    args = parser.parse_args()

    main()
