import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def get_data(filename):
    all_simulations = []
    with open(filename, "r") as t:
        lines = t.readlines()
        simu = []
        for i, l in enumerate(lines):
            if l[:3] == "NEW":
                if i > 0:
                    all_simulations.append(simu)
                    simu = []
            else:
                iteration, pol_reward, expert_reward, trpo_loss,reward_net_loss = l[:-1].split(',')
                simu.append([int(iteration), float(pol_reward), float(expert_reward), float(trpo_loss), float(reward_net_loss)])
        all_simulations.append(simu)

    return np.array(all_simulations)


def to_plot(data, label = None, color=""):
    mean_data = np.mean(data, axis = 0)
    std_data = np.std(data, axis = 0)
    if color == "":
        plt.plot(mean_data, label = label)
        plt.fill_between(np.arange(len(mean_data)), mean_data + std_data, mean_data - std_data, alpha = 0.1)
    elif label!= "EXPERT":
        plt.plot(mean_data,label = label, color=color)
        plt.fill_between(np.arange(len(mean_data)), mean_data + std_data, mean_data - std_data, alpha = 0.1)
    else:
        plt.plot(mean_data,label = label, color=color,  linestyle='--')
    

def main(env_name, path_save_exp=''):
    path_plot = path_save_exp+"/plot"
    os.mkdir(path_plot)

    aeirl_data = get_data(path_save_exp+"/log/aeirl.txt")
    gail_data = get_data(path_save_exp+"/log/gail.txt")

    to_plot(aeirl_data[:,:,2], label = "EXPERT", color="aqua")
    to_plot(aeirl_data[:,:,1], label = "AEIRL", color="blue")
    to_plot(gail_data[:,:,1], label = "GAIL", color="yellow")
    

    plt.xlabel("episode")
    plt.ylabel("episode reward")
    plt.title("Reward Evolution : "+env_name)
    plt.legend()
    plt.savefig(path_plot+"/all_reward_evolution_"+env_name+".png")
    plt.clf()

    to_plot(aeirl_data[:,:,3], label = "AEIRL", color="blue")
    to_plot(gail_data[:,:,3], label = "GAIL", color="yellow")
    

    plt.xlabel("episode")
    plt.ylabel("episode Loss")
    plt.title("TRPO Loss Evolution : "+env_name)
    plt.legend()
    plt.grid()
    plt.savefig(path_plot+"/trpo_loss_"+env_name+".png")
    plt.clf()

    to_plot(aeirl_data[:,:,4], label = "AEIRL", color="blue")

    plt.xlabel("episode")
    plt.ylabel("episode Loss")
    plt.title("Reward Network (Auto-Encoder) Loss Evolution : "+env_name)
    plt.legend()
    plt.grid()
    plt.savefig(path_plot+"/auto_encoder_loss_"+env_name+".png")
    plt.clf()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    parser.add_argument(
        "--folder_file",
        type=str,
        default="",
        help="Need log Path"
    )
    args = parser.parse_args()

    main(args.env_name, args.folder_file)