"""DQN trainer for FlappyBird."""
from ray.rllib.algorithms.dqn import DQNConfig
from .base_trainer import BaseTrainer
from environment import register_environment
import config


class DQNTrainer(BaseTrainer):
    """Trainer for DQN algorithm."""

    def __init__(self, num_workers: int = 1):  # Only 1 worker for DQN
        super().__init__("DQN")
        self.num_workers = num_workers
        register_environment()
        self.build_algo()

    def build_algo(self):
        """Build the DQN algorithm."""
        dqn_config = (
            DQNConfig()
            .environment(env="flappy-bird")
            .env_runners(
                num_env_runners=self.num_workers,
                num_cpus_per_env_runner=1,  # This goes in env_runners, not resources
            )
            .training(
                lr=config.DQN_CONFIG["lr"],
                train_batch_size_per_learner=config.DQN_CONFIG["train_batch_size_per_learner"],
                num_steps_sampled_before_learning_starts=config.DQN_CONFIG["num_steps_sampled_before_learning_starts"],
                target_network_update_freq=config.DQN_CONFIG["target_network_update_freq"],
            )
            .resources(
                num_gpus=0,
            )
        )

        self.algo = dqn_config.build_algo()
        print(f"DQN algorithm initialized with {self.num_workers} workers")