import argparse
import json
from tqdm import tqdm

from models.aeirl import AEIRL
from models.gail import GAIL
from models.nets import Discriminator, AE
from stable_baselines3 import PPO
import gym
import torch
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

def tsne(datax, color, perplexity_list=[5, 10, 30, 50, 100]):
    
    tsne_perp = [ TSNE(n_components=2, random_state=0, perplexity=perp) for perp in perplexity_list]
    tsne_perp_data = [ tsne.fit_transform(datax) for tsne in tsne_perp]
    
    plt.figure(figsize=(15,15))
    for i in tqdm(range(len(perplexity_list))):
        plt.subplot(3,2, i+1)
        plt.title(f"{args.env_name} - {args.model} - Perplexity : " + str(perplexity_list[i]))
        plt.scatter(tsne_perp_data[i][0,0],tsne_perp_data[i][1,1], c="skyblue", label="Expert Sample")
        plt.scatter(tsne_perp_data[i][1:-1,0],tsne_perp_data[i][1:-1,1], c=np.where(color==0,"skyblue", "orchid")[1:-1], s=np.where(color==0,30, 15)[1:-1], alpha=np.where(color==0,1, 1)[1:-1])
        plt.scatter(tsne_perp_data[i][-1:,0],tsne_perp_data[i][-1:,1], c="orchid", label="Policy Sample")

    plt.legend()
    plt.subplot(3,2, i+2)
    plt.axis('off')
    plt.show()

def get_act(pi, state, deterministic=False):
    pi.eval()
    state = FloatTensor(state)
    if deterministic:
        return pi(state, deterministic=True).detach().cpu().numpy()

    distb = pi(state)

    action = distb.sample().detach().cpu().numpy()

    return action

def get_sample(env, model, num_steps_per_iter, nb_eval=10, nb_step_eval=10000, noise=0, horizon=None, render=False, expert=True):

        rwd_iter = []

        obs = []
        acts = []

        steps = 0

############ EXPERT DATA #######################################################
        while steps < num_steps_per_iter:
            ep_obs = []
            ep_rwds = []

            t = 0
            done = False

            ob = env.reset()

            while not done and steps < num_steps_per_iter:
                if env.unwrapped.spec.id in ["Hopper-v2", "Swimmer-v2", "Walker2d-v2", "Reacher-v2"] and expert:
                    act = model.predict(ob)[0]
                else:
                    act = get_act(model, ob)

                ep_obs.append(ob)
                obs.append(ob)
                acts.append(act)

                if render:
                    env.render()
                ob, rwd, done, info = env.step(act)

                ep_rwds.append(rwd)

                t += 1
                steps += 1

                if horizon is not None:
                    if t >= horizon:
                        done = True
                        break

            if done:
                rwd_iter.append(np.sum(ep_rwds))

            ep_obs = FloatTensor(np.array(ep_obs))
            ep_rwds = FloatTensor(ep_rwds)

        rwd_mean = np.mean(rwd_iter)
        # print(
        #     "Expert Reward Mean: {}".format(rwd_mean)
        # )

        obs = FloatTensor(np.array(obs))
        acts = FloatTensor(np.array(acts))

        return obs, acts, rwd_mean

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

    if args.noisy:
        noisy = "_0.3_"
    else:
        noisy = "_0_"
    if args.model=="aeirl":
        model = AEIRL
        reward_model = AE
        reward_net = reward_model(state_dim, action_dim, False)
        reward_net.load_state_dict(torch.load(f"final_reward_nets/{args.env_name[:-3]}/aeirl_autoencoder{noisy}.ckpt"))
    elif args.model=="gail":
        model = GAIL
        reward_model = Discriminator
        reward_net = reward_model(state_dim, action_dim, False)
        reward_net.load_state_dict(torch.load(f"final_reward_nets/{args.env_name[:-3]}/gail_discriminator{noisy}.ckpt"))
    else:
        print("Wrong model")
        return 

    pol_non_noisy = model(state_dim, action_dim, False)
    pol_non_noisy.pi.load_state_dict(torch.load(path_non_noisy))

    pol_noisy = model(state_dim, action_dim, False)
    pol_noisy.pi.load_state_dict(torch.load(path_noisy))


    expert_policy = PPO.load(f"experts/{args.env_name}/PPO-{args.env_name}")

    obs, acts, rwd_mean = get_sample(env, pol_noisy.pi, config["num_steps_per_iter"], nb_eval=10, nb_step_eval=5000, noise=0, horizon=None, render=False, expert=False)
    exp_obs, exp_acts, exp_rwd_mean = get_sample(env, expert_policy, config["num_steps_per_iter"], nb_eval=10, nb_step_eval=5000, noise=0, horizon=None, render=False)

    res = reward_net.get_first_linear(obs, acts).cpu().numpy()
    exp_res = reward_net.get_first_linear(exp_obs, exp_acts).cpu().numpy()

    color = np.concatenate((np.zeros(np.shape(exp_res), dtype=int),np.ones(np.shape(res), dtype=int)))[:,0]

    print("Start t-sne...")
    
    tsne(np.concatenate((exp_res,res)), color)

    print("End run")



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
        help="Number total of epoch"
    )

    parser.add_argument(
        "--noisy",
        type=bool,
        default=False,
        help="Data with noisy or not"
    )

    args = parser.parse_args()

    main()
