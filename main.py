import argparse
from tqdm import tqdm

from train import main as main_aeirl
from train_aeirl import main as main_gail
from plot_from_log import main as plot

def main(env_name, nb_run):

    print(f"Start AEIRL ... ")
    for i in tqdm(range(nb_run)):
        main_aeirl(env_name)

    print(f"Start GAIL ... ")
    for i in tqdm(range(nb_run)):
        main_gail(env_name)
    
    print("End Training phase")
    print("Plot...")
    plot(env_name)

    print("Plot saved")

    print("End.")
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
        "--nb_run",
        type=int,
        default=10,
        help="Number of run time"
    )
    args = parser.parse_args()

    main(args.env_name, args.nb_run)
