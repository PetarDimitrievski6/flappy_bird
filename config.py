"""Configuration settings for training."""

DEFAULT_NUM_WORKERS = 4
DEFAULT_NUM_ITERATIONS = 2000
DEFAULT_TEST_EPISODES = 10
DEFAULT_LOG_ACTION_DISTRIBUTION = True
TOP_TOUCH_PENALTY = -4.0

# Episode limits
MAX_EPISODE_STEPS = 10000
MAX_STEPS_BONUS_REWARD = 1000.0

# Early stopping settings (disabled by default)
EARLY_STOPPING_ENABLED = False
EARLY_STOPPING_REWARD_THRESHOLD = 500.0
EARLY_STOPPING_PATIENCE = 50

# PPO-specific learning rate schedule (linear decay)
LR_SCHEDULE_ENABLED = True
LR_SCHEDULE_END_FACTOR = 0.1

PPO_CONFIG = {
    "lr": 0.00025,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_param": 0.2,
    "vf_clip_param": 10.0,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.02,
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    "grad_clip": 0.5,
    "train_batch_size_per_learner": 8000,
    "minibatch_size": 256,
    "num_epochs": 10,
    "rollout_fragment_length": 100,
    "num_envs_per_worker": 5,
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "tanh",
        "vf_share_layers": True,
    },
}

DQN_CONFIG = {
    "lr": 5e-4,
    "gamma": 0.99,
    "train_batch_size_per_learner": 512,
    "num_steps_sampled_before_learning_starts": 2000,
    "target_network_update_freq": 2000,
    "double_q": True,
    "dueling": True,
    "n_step": 3,
    "replay_buffer_config": {
        "type": "PrioritizedEpisodeReplayBuffer",
        "capacity": 100_000,
        "alpha": 0.6,
        "beta": 0.4,
    },
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "epsilon_timesteps": 100_000,
    },
    "model": {
        "fcnet_hiddens": [256, 128],
        "fcnet_activation": "relu",
    },
}

# CQL uses old API stack (enable_rl_module_and_learner=False)
CQL_CONFIG = {
    "lr": 5e-4,
    "gamma": 0.99,
    "train_batch_size": 512,
    "train_batch_size_per_learner": 512,
    "num_steps_sampled_before_learning_starts": 2000,
    "target_network_update_freq": 2000,
    "double_q": True,
    "dueling": True,
    "n_step": 3,
    "replay_buffer_config": {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 100_000,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "prioritized_replay_eps": 1e-6,
    },
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "epsilon_timesteps": 100_000,
    },
    "cql_min_q_weight": 1.0,
    "cql_temperature": 1.0,
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    },
}

# Rainbow DQN combines: Double DQN, Dueling, Multi-step, Noisy Nets,
# Distributional (C51), Prioritized Replay.
# Noisy nets replace epsilon-greedy exploration entirely.
# v_min/v_max cover the per-step reward range (not total episode return).
RAINBOW_DQN_CONFIG = {
    "lr": 6.25e-4,
    "gamma": 0.99,
    "train_batch_size_per_learner": 256,
    "double_q": True,
    "dueling": True,
    "n_step": 3,
    "noisy": True,
    "num_atoms": 51,
    "v_min": -10.0,
    "v_max": 50.0,
    "num_steps_sampled_before_learning_starts": 1000,
    "target_network_update_freq": 500,
    "tau": 1.0,
    "replay_buffer_capacity": 100_000,
    "replay_alpha": 0.6,
    "replay_beta": 0.4,
    "model": {
        "fcnet_hiddens": [256, 128],
        "fcnet_activation": "relu",
    },
}

IMPALA_CONFIG = {
    "lr": 5e-4,
    "gamma": 0.99,
    "vtrace": True,
    "vtrace_clip_rho_threshold": 1.0,
    "vtrace_clip_pg_rho_threshold": 1.0,
    "train_batch_size_per_learner": 4000,
    "minibatch_size": 500,
    "num_epochs": 3,
    "rollout_fragment_length": 100,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "grad_clip": 5.0,
    "learner_queue_size": 8,
    "num_gpu_loader_threads": 2,
    "broadcast_interval": 1,
    "min_time_s_per_iteration": 1,
    "num_envs_per_worker": 5,
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "vf_share_layers": True,
    },
}

APPO_CONFIG = {
    "lr": 0.0003,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_param": 0.2,
    "vf_clip_param": 10.0,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "grad_clip": 1.0,
    "vtrace": True,
    "train_batch_size_per_learner": 4000,
    "minibatch_size": 500,
    "num_epochs": 10,
    "rollout_fragment_length": 100,
    "num_envs_per_worker": 5,
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "vf_share_layers": True,
    },
}
