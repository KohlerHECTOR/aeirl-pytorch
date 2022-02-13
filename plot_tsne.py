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
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

def get_tse_data(res, exp_res, color, perplexity_list):
    datax = np.concatenate((exp_res,res))
    tsne_perp = [TSNE(n_components=2, init ='pca', learning_rate = 200.0, perplexity=per, n_jobs=-1) for per in perplexity_list]
    tsne_perp_data = [tsne.fit_transform(datax) for tsne in tsne_perp]
    
    return tsne_perp_data

def plot_tsne(tsne_perp_data, tsne_perp_data2, tsne_perp_data3, tsne_perp_data4, color1, color2, color3, color4, perplexity_list):
    expert_color = "forestgreen"
    aeirl_color = "deepskyblue"
    gail_color = "tomato"
    scatter_size = 5
    plt.figure(figsize=(15,15))
    sub_i = 1
    for i in range(len(perplexity_list)):

        #AEIRL
        plt.subplot(6,6, sub_i)
        plt.axis('off')
        sub_i+=1
        if i==0:
            plt.title("Non-Noisy Data")
        plt.scatter(tsne_perp_data[i][5000:-1,0],tsne_perp_data[i][5000:-1,1], c=np.where(color1==0,expert_color, aeirl_color)[5000:-1], s=np.where(color1==0,scatter_size, scatter_size)[5000:-1], alpha=np.where(color1==0,1, 1)[5000:-1])
        plt.scatter(tsne_perp_data[i][-1:,0],tsne_perp_data[i][-1:,1], c=aeirl_color, label="AEIRL Sample")
        
        plt.subplot(6,6, sub_i)
        plt.axis('off')
        sub_i+=1
        if i==0:
            plt.title("Non-Noisy Data")
        plt.scatter(tsne_perp_data[i][0,0],tsne_perp_data[i][1,1], c=expert_color, label="Expert Sample")
        plt.scatter(tsne_perp_data[i][1:-1,0],tsne_perp_data[i][1:-1,1], c=np.where(color1==0,expert_color, aeirl_color)[1:-1], s=np.where(color1==0,scatter_size, scatter_size)[1:-1], alpha=np.where(color1==0,1, 0.1)[1:-1])
        plt.scatter(tsne_perp_data[i][-1:,0],tsne_perp_data[i][-1:,1], c=aeirl_color, label="AEIRL Sample")
        if i==0:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=3, fancybox=True, shadow=True)
            
        plt.subplot(6,6, sub_i)
        plt.axis('off')
        sub_i+=1
        if i==0:
            plt.title("Non-Noisy Data")
        plt.scatter(tsne_perp_data[i][0,0],tsne_perp_data[i][1,1], c=expert_color)
        plt.scatter(tsne_perp_data[i][1:5000,0],tsne_perp_data[i][1:5000,1], c=np.where(color1==0,expert_color, aeirl_color)[1:5000], s=np.where(color1==0,scatter_size, scatter_size)[1:5000], alpha=np.where(color1==0,1, 1)[1:5000])
        plt.scatter(0,0, alpha=0, label=f"{args.env_name}: Perplexity={str(perplexity_list)}        ")
        if i==0:
            plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1.75),ncol=3)

        #GAIL  
        plt.subplot(6,6, sub_i)
        plt.axis('off')
        sub_i+=1
        if i==0:
            plt.title("Non-Noisy Data")
        plt.scatter(tsne_perp_data2[i][0,0],tsne_perp_data2[i][1,1], c=expert_color, label="Expert Sample")
        plt.scatter(tsne_perp_data2[i][1:5000,0],tsne_perp_data2[i][1:5000,1], c=np.where(color2==0,expert_color, gail_color)[1:5000], s=np.where(color2==0,scatter_size, scatter_size)[1:5000], alpha=np.where(color2==0,1, 1)[1:5000])

        plt.subplot(6,6, sub_i)
        plt.axis('off')
        
        sub_i+=1
        if i==0:
            plt.title("Non-Noisy Data")
        plt.scatter(tsne_perp_data2[i][0,0],tsne_perp_data2[i][1,1], c=expert_color, label="Expert Sample")
        plt.scatter(tsne_perp_data2[i][1:-1,0],tsne_perp_data2[i][1:-1,1], c=np.where(color2==0,expert_color, gail_color)[1:-1], s=np.where(color2==0,scatter_size, scatter_size)[1:-1], alpha=np.where(color2==0,1, 0.1)[1:-1])
        plt.scatter(tsne_perp_data2[i][-1:,0],tsne_perp_data2[i][-1:,1], c=gail_color, label="GAIL Sample")
        if i==0:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),ncol=3, fancybox=True, shadow=True)
            
        plt.subplot(6,6, sub_i)
        plt.axis('off')
        sub_i+=1
        if i==0:
            plt.title("Non-Noisy Data")
        plt.scatter(tsne_perp_data2[i][5000:-1,0],tsne_perp_data2[i][5000:-1,1], c=np.where(color2==0,expert_color, gail_color)[5000:-1], s=np.where(color2==0,scatter_size, scatter_size)[5000:-1], alpha=np.where(color2==0,1, 1)[5000:-1])
        plt.scatter(tsne_perp_data2[i][-1:,0],tsne_perp_data2[i][-1:,1], c=gail_color, label="GAIL Sample")
    
    for i in range(len(perplexity_list)):
        
        #AEIRL noisy
        plt.subplot(6,6, sub_i)
        plt.axis('off')
        sub_i+=1
        if i==0:
            plt.title("Noisy Data")
        plt.scatter(tsne_perp_data3[i][5000:-1,0],tsne_perp_data3[i][5000:-1,1], c=np.where(color3==0,expert_color, aeirl_color)[5000:-1], s=np.where(color3==0,scatter_size, scatter_size)[5000:-1], alpha=np.where(color3==0,1, 1)[5000:-1])
        plt.scatter(tsne_perp_data3[i][-1:,0],tsne_perp_data3[i][-1:,1], c=aeirl_color, label="AEIRL Sample")
        
        plt.subplot(6,6, sub_i)
        plt.axis('off')
        if i==0:
            plt.title("Noisy Data")
        sub_i+=1
        plt.scatter(tsne_perp_data3[i][0,0],tsne_perp_data3[i][1,1], c=expert_color, label="Expert Sample")
        plt.scatter(tsne_perp_data3[i][1:-1,0],tsne_perp_data3[i][1:-1,1], c=np.where(color3==0,expert_color, aeirl_color)[1:-1], s=np.where(color3==0,scatter_size, scatter_size)[1:-1], alpha=np.where(color3==0,1, 0.1)[1:-1])
        plt.scatter(tsne_perp_data3[i][-1:,0],tsne_perp_data3[i][-1:,1], c=aeirl_color, label="AEIRL Sample")
            
        plt.subplot(6,6, sub_i)
        plt.axis('off')
        if i==0:
            plt.title("Noisy Data")
        sub_i+=1
        plt.scatter(tsne_perp_data3[i][0,0],tsne_perp_data3[i][1,1], c=expert_color, label="Expert Sample")
        plt.scatter(tsne_perp_data3[i][1:5000,0],tsne_perp_data3[i][1:5000,1], c=np.where(color3==0,expert_color, aeirl_color)[1:5000], s=np.where(color3==0,scatter_size, scatter_size)[1:5000], alpha=np.where(color3==0,1, 1)[1:5000])
         
        #GAIL noisy
        plt.subplot(6,6, sub_i)
        plt.axis('off')
        sub_i+=1
        if i==0:
            plt.title("Noisy Data")
        plt.scatter(tsne_perp_data4[i][0,0],tsne_perp_data4[i][1,1], c=expert_color, label="Expert Sample")
        plt.scatter(tsne_perp_data4[i][1:5000,0],tsne_perp_data4[i][1:5000,1], c=np.where(color4==0,expert_color, gail_color)[1:5000], s=np.where(color4==0,scatter_size, scatter_size)[1:5000], alpha=np.where(color4==0,1, 1)[1:5000])

        plt.subplot(6,6, sub_i)
        plt.axis('off')
        if i==0:
            plt.title("Noisy Data")
        sub_i+=1
        plt.scatter(tsne_perp_data4[i][0,0],tsne_perp_data4[i][1,1], c=expert_color, label="Expert Sample")
        plt.scatter(tsne_perp_data4[i][1:-1,0],tsne_perp_data4[i][1:-1,1], c=np.where(color4==0,expert_color, gail_color)[1:-1], s=np.where(color4==0,scatter_size, scatter_size)[1:-1], alpha=np.where(color4==0,1, 0.1)[1:-1])
        plt.scatter(tsne_perp_data4[i][-1:,0],tsne_perp_data4[i][-1:,1], c=gail_color, label="GAIL Sample")
            
        plt.subplot(6,6, sub_i)
        plt.axis('off')
        if i==0:
            plt.title("Noisy Data")
        sub_i+=1
        plt.scatter(tsne_perp_data4[i][5000:-1,0],tsne_perp_data4[i][5000:-1,1], c=np.where(color4==0,expert_color, gail_color)[5000:-1], s=np.where(color4==0,scatter_size, scatter_size)[5000:-1], alpha=np.where(color4==0,1, 1)[5000:-1])
        plt.scatter(tsne_perp_data4[i][-1:,0],tsne_perp_data4[i][-1:,1], c=gail_color, label="GAIL Sample")

    plt.savefig('tsne.pdf')
    plt.show()
    
def get_data(model, noisy):
    with open("config.json") as f:
        config = json.load(f)[args.env_name]

    env = gym.make(args.env_name)

    list_model_to_test = []

    state_dim = len(env.observation_space.high)
    action_dim = env.action_space.shape[0]

    list_name = [model+' non noisy', model+' noisy']

    path_non_noisy = f"final_policies/{args.env_name[:-3]}/{model}_policy_0_.ckpt"

    path_noisy = f"final_policies/{args.env_name[:-3]}/{model}_policy_0.3_.ckpt"

    if noisy:
        noisy = "_0.3_"
    else:
        noisy = "_0_"
    if model=="aeirl":
        model = AEIRL
        reward_model = AE
        reward_net = reward_model(state_dim, action_dim, False)
        reward_net.load_state_dict(torch.load(f"final_reward_nets/{args.env_name[:-3]}/aeirl_autoencoder{noisy}.ckpt"))
    elif model=="gail":
        model = GAIL
        reward_model = Discriminator
        reward_net = reward_model(state_dim, action_dim, False)
        reward_net.load_state_dict(torch.load(f"final_reward_nets/{args.env_name[:-3]}/gail_discriminator{noisy}.ckpt"))
    else:
        print("Wrong model")

    pol_non_noisy = model(state_dim, action_dim, False)
    pol_non_noisy.pi.load_state_dict(torch.load(path_non_noisy))

    pol_noisy = model(state_dim, action_dim, False)
    pol_noisy.pi.load_state_dict(torch.load(path_noisy))


    expert_policy = PPO.load(f"experts/{args.env_name}/PPO-{args.env_name}")

    obs, acts, rwd_mean = get_sample(env, pol_noisy.pi, config["num_steps_per_iter"], nb_eval=10, nb_step_eval=5000, noise=0, horizon=None, render=False, expert=False)
    exp_obs, exp_acts, exp_rwd_mean = get_sample(env, expert_policy, config["num_steps_per_iter"], nb_eval=10, nb_step_eval=5000, noise=0, horizon=None, render=False)

    res = reward_net.get_first_linear(obs, acts).cpu().numpy()
    #color = np.zeros(np.shape(res), dtype=int)
    exp_res = reward_net.get_first_linear(exp_obs, exp_acts).cpu().numpy()
    #color = np.concatenate((np.ones(np.shape(exp_res), dtype=int),color))
    color = np.concatenate((np.zeros(np.shape(res), dtype=int),np.ones(np.shape(exp_res), dtype=int)))[:,0]
    #tsne(np.concatenate((exp_res,res)), color)
    #print(res_AE)
    return res, exp_res, color

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
                    act = model.predict(ob, deterministic=True)[0]
                else:
                    act = get_act(model, ob, deterministic=True)

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

def main(perplexity_list = [5, 50]):
    
    res1, exp_res1, color1 = get_data("aeirl", False)
    res2, exp_res2, color2 = get_data("gail", False)

    res3, exp_res3, color3 = get_data("aeirl", True)
    res4, exp_res4, color4 = get_data("gail", True)
    print("Start t-sne...")

    tsne_perp_data1 = get_tse_data(res1, exp_res1, color1, perplexity_list)
    tsne_perp_data2 = get_tse_data(res2, exp_res2, color2, perplexity_list)
    tsne_perp_data3 = get_tse_data(res3, exp_res3, color3, perplexity_list)
    tsne_perp_data4 = get_tse_data(res4, exp_res4, color4, perplexity_list)

    plot_tsne(tsne_perp_data1, tsne_perp_data2, tsne_perp_data3, tsne_perp_data4, color1, color2, color3, color4, perplexity_list)


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

    args = parser.parse_args()

    main()
