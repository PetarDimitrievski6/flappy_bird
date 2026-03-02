"""DQN trainer for FlappyBird."""
from ray.rllib.algorithms.dqn import DQNConfig
from .base_trainer import BaseTrainer
from environment import register_environment
import config


class DQNTrainer(BaseTrainer):
    def __init__(self, num_workers: int = 1):
        super().__init__("DQN")
        self.num_workers = num_workers
        register_environment()
        self.build_algo()

    def build_algo(self):
        num_gpus_per_learner, device_msg = self._setup_gpu()

        exploration_config = config.DQN_CONFIG.get("exploration_config", {})
        epsilon_schedule = None
        if exploration_config.get("type") == "EpsilonGreedy":
            epsilon_schedule = [
                (0, exploration_config.get("initial_epsilon", 1.0)),
                (exploration_config.get("epsilon_timesteps", 0), exploration_config.get("final_epsilon", 0.05)),
            ]

        training_kwargs = {
            "lr": config.DQN_CONFIG["lr"],
            "gamma": config.DQN_CONFIG["gamma"],
            "train_batch_size_per_learner": config.DQN_CONFIG["train_batch_size_per_learner"],
            "train_batch_size": config.DQN_CONFIG["train_batch_size_per_learner"],
            "num_steps_sampled_before_learning_starts": config.DQN_CONFIG["num_steps_sampled_before_learning_starts"],
            "target_network_update_freq": config.DQN_CONFIG["target_network_update_freq"],
            "replay_buffer_config": config.DQN_CONFIG["replay_buffer_config"],
            "double_q": config.DQN_CONFIG["double_q"],
            "dueling": config.DQN_CONFIG["dueling"],
            "n_step": config.DQN_CONFIG["n_step"],
            "model": config.DQN_CONFIG["model"],
        }
        if epsilon_schedule is not None:
            training_kwargs["epsilon"] = epsilon_schedule

        dqn_config = (
            DQNConfig()
            .environment(env="flappy-bird")
            .env_runners(
                num_env_runners=self.num_workers,
                num_cpus_per_env_runner=1,
            )
            .learners(
                num_learners=0,
                num_gpus_per_learner=num_gpus_per_learner,
            )
            .training(**training_kwargs)
            .framework("torch")
        )

        self.algo = dqn_config.build_algo()
        print(f"DQN initialized with {self.num_workers} workers on {device_msg}")
