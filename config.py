"""Configuration settings for training."""

# Training configuration
DEFAULT_NUM_WORKERS = 2
DEFAULT_NUM_ITERATIONS = 500
DEFAULT_TEST_EPISODES = 10

# PPO specific config
PPO_CONFIG = {
    "lr": 0.0001,  # slightly smaller for stability
    "gamma": 0.99,
    "lambda": 0.95,  # GAE
    "clip_param": 0.2,  # PPO clipping
    "vf_clip_param": 10.0,
    "vf_loss_coeff": 1.0,
    "entropy_coeff": 0.01,  # encourage exploration
    "num_sgd_iter": 20,  # train more per batch
    "sgd_minibatch_size": 512,  # larger minibatch
    "train_batch_size": 8000,  # collect bigger batches
    "train_batch_size_per_learner": 4000,
    "num_epochs": 20,  # more epochs per batch
    "rollout_fragment_length": 200,  # can keep 'auto'
    "model": {
        "fcnet_hiddens": [512, 512],  # bigger network
        "fcnet_activation": "tanh",
    },
    "framework": "torch",
    "num_workers": 2,  # can keep 2
    "num_envs_per_worker": 4,  # vectorized for faster learning
    "log_level": "WARN",
    "clip_actions": False,
    "normalize_actions": True,
    "explore": True,
}
DQN_CONFIG = {
    "lr": 5e-4,
    "gamma": 0.99,
    "train_batch_size_per_learner": 128,
    "num_steps_sampled_before_learning_starts": 2000,
    "target_network_update_freq": 2000,
    "replay_buffer_config": {
        "type": "PrioritizedEpisodeReplayBuffer",
        "capacity": 100_000,
        "alpha": 0.6,
        "beta": 0.4,
    },
    "double_q": True,
    "dueling": True,
    "n_step": 3,
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "epsilon_timesteps": 100_000,
    },
    "model": {
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
    },
}