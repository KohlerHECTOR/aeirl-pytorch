import argparse
import json

from models.aeirl import AEIRL
from models.gail import GAIL
import gym
import torch
import numpy as np
from tqdm import tqdm

def eval_pol(env, model, nb_eval=10, nb_step_eval=10000, render=False):

    eval = []
    for n in tqdm(range(nb_eval)):
        env.seed(n)
        s = env.reset()
        reward = 0

        for t in range(nb_step_eval):
            if render:
                env.render()

            action = model.act(s, deterministic=True)

            next_state, r, done, _ = env.step(action)

            s = next_state
            reward += r

            if done:
                break


        eval.append(reward)

    return eval

def main():
    
    with open("config.json") as f:
        config = json.load(f)[args.env_name]

    env = gym.make(args.env_name)

    list_model_to_test = []

    state_dim = len(env.observation_space.high)
    action_dim = env.action_space.shape[0]

    list_name = [args.model+' non noisy', args.model+' noisy']

    if args.path_non_noisy != None:
        path_non_noisy = args.path_non_noisy
    else:
        path_non_noisy = f"final_policies/{args.env_name[:-3]}/{args.model}_policy_0_.ckpt"
    
    if args.path_noisy != None:
        path_noisy = args.path_noisy
    else:
        path_noisy = f"final_policies/{args.env_name[:-3]}/{args.model}_policy_0.3_.ckpt"

    pol_non_noisy = AEIRL(state_dim, action_dim, False)
    pol_non_noisy.pi.load_state_dict(torch.load(path_non_noisy))

    list_model_to_test.append(pol_non_noisy)

    pol_noisy = AEIRL(state_dim, action_dim, False)
    pol_noisy.pi.load_state_dict(torch.load(path_noisy))

    list_model_to_test.append(pol_noisy)

    list_res = []
    res_msg = ""
    for i, model in enumerate(list_model_to_test):
        res = eval_pol(env, model, nb_eval=args.nb_eval, nb_step_eval=config['nb_step_eval'], render=args.render)
        list_res.append(np.mean(res))
        res_msg +=f"{list_name[i]} : {np.mean(res)} (+/-) {np.std(res)}\n"

    
    print(res_msg)
    print(f"Pourcentage de Moyenne de détérioration : { round((list_res[0]-list_res[1])/abs(list_res[0]), 4)*100}%")

    
    #print(res_AE)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="Walker2d-v2",
        help="Type the environment name to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="aeirl",
        help="Type of policy to eval : [aeirl, gail]"
    )

    parser.add_argument(
        "--path-non-noisy",
        type=str,
        default=None,
        help="Full path of the policy non noisy"
    )

    parser.add_argument(
        "--path-noisy",
        type=str,
        default=None,
        help="Full path of the policy noisy"
    )
    parser.add_argument(
        "--render",
        type=bool,
        default=False,
        help="Full path of the policy noisy"
    )

    parser.add_argument(
        "--nb_eval",
        type=int,
        default=100,
        help="Type the environment name to run"
    )
    args = parser.parse_args()

    main()
