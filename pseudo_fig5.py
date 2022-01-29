import AE
import Discriminator
import PolicyNet
nb_exp = 10
env = gym.make("env")
reward_AE = AE()
reward_AE.load("AE_apres_AEIRL_sur_ENV.pth")

reward_Discr = Discriminator()
reward_Discr.load("Discr_apres_GAIL_sur_ENV.pth")

ppo_difs_levels_path = PATH

all_exps_AE = []
all_exps_Discr = []
for exp in range(nb_exp)
    exp_results_AE = []
    exp_results_Discr = []
    traj_length = []
    for ppo_policy in ppo_difs_levels_path:
        policy = PolicyNet()
        policy.load(ppo_policy)
        state = env.reset()
        sum_rewards_AE = 0
        sum_rewards_Discr = 0
        ## evaluate intermediate ppo policy using AE or Discr reward
        steps_done = 0
        for steps in steps_per_traj:

            action = policy.select_action(state)
            next_state, _, done, info = env.step(action)
            steps_done += 1
            sum_rewards_AE += 1/ (1 + reward_AE(state, action))
            sum_rewards_Discr += log(reward_Discr(state, action))

            if done:
                break
            state = next_state

        traj_length.append(steps_done)
        exp_results_AE.append(sum_rewards_AE)
        exp_results_Discr.append(sum_rewards_Discr)

    for i in range(len(exp_results_AE)):
        ### TODO: scaled_normalized
        exp_results_AE[i] = scaled_normalized(exp_results_AE[i], traj_length)
        exp_results_Discr[i] = scaled_normalized(exp_results_Discr[i], traj_length)

    all_exps_AE.append(exp_results_AE)
    all_exps_Discr.append(exp_results_Discr)


def to_plot(data, label = None):
    mean_data = np.mean(data, axis = 0)
    std_data = np.std(data, axis = 0)
    plt.plot(mean_data, label = label)
    plt.fill_between(np.arange(len(mean_data)), mean_data + std_data, mean_data - std_data, alpha = 0.1)


to_plot(all_exps_AE, label = "AEIRL")
to_plot(all_exps_Discr, label = "GAIL")
