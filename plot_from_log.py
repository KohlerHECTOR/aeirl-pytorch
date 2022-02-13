import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as intplt
import argparse
import json
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
                iteration, expert_reward, trpo_loss,reward_net_loss = l[:-1].split(',')
                simu.append([int(iteration), float(expert_reward), float(trpo_loss), float(reward_net_loss)])
        all_simulations.append(simu)

    return np.array(all_simulations)

def get_eval_data(filename):
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
                iteration, eval_reward = l[:-1].split(',')
                simu.append([int(iteration), float(eval_reward)])
        all_simulations.append(simu)

    return np.array(all_simulations)

def to_plot(data, label = None, color="", smooth = False):
    mean_data = np.mean(data, axis = 0)
    if smooth:
        x_new = np.linspace(0,25, 5000)
        a_BSpline = intplt.make_interp_spline(np.arange(len(mean_data)), mean_data)
        y_new = a_BSpline(x_new)

    std_data = np.std(data, axis = 0)
    if color == "":
        plt.plot(mean_data, label = label)
        plt.fill_between(np.arange(len(mean_data)), mean_data + std_data, mean_data - std_data, alpha = 0.05)
    elif label!= "EXPERT":
        if smooth:
            plt.plot(x_new, y_new, label = label, color=color)
        else:
            plt.plot(mean_data,label = label, color=color)
        plt.fill_between(np.arange(len(mean_data)), mean_data + std_data, mean_data - std_data, alpha = 0.05)
    else:
        plt.plot(mean_data,label = label, color=color,  linestyle='--')


def main(env_name, path_save_exp='', noise = 0):



    path_plot = path_save_exp+"/plot"
    try:
        os.mkdir(path_plot)
    except FileExistsError:
        print("ok")

    aeirl_data = get_data(path_save_exp+"/log/aeirl.txt")
    gail_data = get_data(path_save_exp+"/log/gail.txt")
    aeirl_eval_data = get_eval_data(path_save_exp+"/log/aeirl_eval.txt")
    gail_eval_data = get_eval_data(path_save_exp+"/log/gail_eval.txt")

    to_plot(aeirl_data[:,:26,1], label = "EXPERT", color="black")
    to_plot(aeirl_eval_data[:,:,1], label = "AEIRL", color="blue")
    to_plot(gail_eval_data[:,:,1], label = "GAIL", color="red")


    with open("config.json") as f:
        config = json.load(f)[env_name]

    tot_steps = config["num_iters"] * config["num_steps_per_iter"]
    tot_evals = config["num_iters"] / config["eval_freq"]
    steps_per_eval_in_thousands = tot_steps / tot_evals / 1e3
    num_steps_per_iter_in_thousands = config["num_steps_per_iter"] / 1e3

    plt.xlabel("x" + str(int(steps_per_eval_in_thousands)) +  "K Timesteps")
    plt.ylabel("Eval Reward Sum")
    plt.title(env_name + " , noise = " + str(noise))
    plt.legend()
    plt.grid()
    plt.savefig(path_plot+"/all_reward_evolution_"+env_name+".pdf")
    plt.clf()

    to_plot(aeirl_data[:,:,2], label = "AEIRL", color="blue")
    to_plot(gail_data[:,:,2], label = "GAIL", color="red")


    plt.xlabel("x" + str(int(num_steps_per_iter_in_thousands)) +  "K Timesteps")
    plt.ylabel("TRPO Loss")
    plt.title(env_name + " , noise = " + str(noise))
    plt.legend()
    plt.grid()
    plt.savefig(path_plot+"/trpo_loss_"+env_name+".pdf")
    plt.clf()

    to_plot(aeirl_data[:,:,3], label = "AEIRL", color="blue")

    plt.xlabel("x" + str(int(num_steps_per_iter_in_thousands)) +  "K Timesteps")
    plt.ylabel("Reward Network Loss")
    plt.title(env_name + " , noise = " + str(noise))
    plt.legend()
    plt.grid()
    plt.savefig(path_plot+"/auto_encoder_loss_"+env_name+".pdf")
    plt.clf()

def main_noisy_on_plot(env_name, path_save_exp='', path_save_exp_noisy = '', noise = 0):


    path_plot = "./plots"
    try:
        os.mkdir(path_plot)
    except FileExistsError:
        print("ok")

    aeirl_data = get_data(path_save_exp+"/log/aeirl.txt")
    gail_data = get_data(path_save_exp+"/log/gail.txt")
    aeirl_eval_data = get_eval_data(path_save_exp+"/log/aeirl_eval.txt")
    gail_eval_data = get_eval_data(path_save_exp+"/log/gail_eval.txt")
    aeirl_data_noisy = get_data(path_save_exp_noisy+"/log/aeirl.txt")
    gail_data_noisy = get_data(path_save_exp_noisy+"/log/gail.txt")
    aeirl_eval_data_noisy = get_eval_data(path_save_exp_noisy+"/log/aeirl_eval.txt")
    gail_eval_data_noisy = get_eval_data(path_save_exp_noisy+"/log/gail_eval.txt")

    to_plot(aeirl_data[:,:26,1], label = "EXPERT", color="black")
    to_plot(aeirl_eval_data[:,:,1], label = "AEIRL", color="blue", smooth = True)
    to_plot(gail_eval_data[:,:,1], label = "GAIL", color="red", smooth = True)
    to_plot(aeirl_eval_data_noisy[:,:,1], label = "AEIRL, " +str(noise) + " noise", color="purple", smooth = True)
    to_plot(gail_eval_data_noisy[:,:,1], label = "GAIL, " +str(noise) + " noise", color="green", smooth = True)


    with open("config.json") as f:
        config = json.load(f)[env_name]

    tot_steps = config["num_iters"] * config["num_steps_per_iter"]
    tot_evals = config["num_iters"] / config["eval_freq"]
    steps_per_eval_in_thousands = tot_steps / tot_evals / 1e3
    num_steps_per_iter_in_thousands = config["num_steps_per_iter"] / 1e3

    plt.xlabel("x" + str(int(steps_per_eval_in_thousands)) +  "K Timesteps")
    plt.ylabel("Eval Reward Sum")
    plt.title(env_name)
    plt.legend()
    plt.grid()
    plt.savefig(path_plot+"/all_reward_evolution_"+env_name+".pdf")
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="Walker2d-v2",
        help="Type the environment name to run. \
            The possible environments are \
                [Hopper-v2, Swimmer-v2, Walker2d-v2]"
    )
    parser.add_argument(
        "--experiment_folder",
        type=str,
        default="",
        help="Need log Path"
    )
    parser.add_argument(
        "--noisy_experiment_folder",
        type=str,
        default="",
        help="Need log Path"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0,
        help="expert data noise"
    )
    args = parser.parse_args()

    main(args.env_name, args.experiment_folder, args.noise)
    # main_noisy_on_plot(args.env_name, args.experiment_folder, args.noisy_experiment_folder, args.noise)
