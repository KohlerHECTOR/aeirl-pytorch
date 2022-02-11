# Training agents

Train a agent with PPO using stable_baselines3:

```console
python model_learning.py
```

## params:

- `env_name` : (str) name of the environnement (Hopper-v2, Reacher-v2, Walker2d-v2...)

- `total_timesteps` : (int) total timesteps (int(1e6),...)

- `render` : (bool) render

- `n_envs` : (int) number of environnement

- `hyperparams_file` : (str) name of the hyperparameter file

- `save_file` : (str) name of the file of the model to be saved

# Evaluate the agent

Evaluate the policy

```console
python model_evaluation.py
```

## params:

- `env_name` : (str) name of the environnement (Hopper-v2, Reacher-v2, Walker2d-v2...)

- `model_file` : (str) file path of the agent to evaluate

- `render` : (bool) render

- `seed` : (int) seed

- `file_path` : (str) path of the tensorboard

- `n_eval_episodes` : (int) the number of episodes to test the agent
