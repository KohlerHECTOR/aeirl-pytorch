import argparse
from tqdm import tqdm
import os
import time
from datetime import date

from train import main as main_gail
from train_aeirl import main as main_aeirl
from plot_from_log import main as plot

from torch import manual_seed

def main(env_name, nb_runs):
    date = time.strftime("%Y,%m,%d,%H,%M,%S").replace(',', '-')

    if not os.path.exists('experiment'):
        os.mkdir('experiment')

    if os.path.exists(date):
        print("The file "+date+" is already created in the 'experiment' file, wait a second or clean the 'experiment' file")
        return

    path_save_exp = 'experiment/'+env_name+'-'+date
    os.mkdir(path_save_exp)

    path_save_log = path_save_exp+'/log'

    os.mkdir(path_save_log)

    print(f"Start AEIRL ... ")
    for i in tqdm(range(nb_runs)):
        manual_seed(i)
        main_aeirl(env_name, path_save_log=path_save_log,  simu_nb = i)

    print(f"Start GAIL ... ")
    for i in tqdm(range(nb_runs)):
        manual_seed(i)
        main_gail(env_name, path_save_log=path_save_log,  simu_nb = i)

    print("End Training phase")
    print("Plot...")
    plot(env_name, path_save_exp)

    print("Plot saved")

    print("End.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="Hopper-v2",
        help="Type the environment name to run. \
            The possible environments are \
                [Hopper-v2, Swimmer-v2, Walker2d-v2]" # [CartPole-v1, Pendulum-v0, BipedalWalker-v3, Hopper-v2, Swimmer-v2, Walker2d-v2]"
    )
    parser.add_argument(
        "--nb_runs",
        type=int,
        default=5,
        help="Number of run time"
    )
    # parser.add_argument('--envs_mujoco', nargs='+',
    #                     default=["Hopper-v2", "Swimmer-v2", "Walker2d-v2"])
    args = parser.parse_args()

    main(**vars(args))
