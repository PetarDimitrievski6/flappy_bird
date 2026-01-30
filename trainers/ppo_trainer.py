"""PPO trainer for FlappyBird."""
from ray.rllib.algorithms.ppo import PPOConfig
from .base_trainer import BaseTrainer
from environment import register_environment
import config


class PPOTrainer(BaseTrainer):
    """Trainer for PPO algorithm."""

    def __init__(self, num_workers: int = config.DEFAULT_NUM_WORKERS):
        super().__init__("PPO")
        self.num_workers = num_workers
        register_environment()
        self.build_algo()

    def build_algo(self):
        """Build the PPO algorithm."""
        ppo_config = (
            PPOConfig()
            .environment(env="flappy-bird")
            .env_runners(num_env_runners=self.num_workers)
            .training(
                lr=config.PPO_CONFIG["lr"],
                train_batch_size_per_learner=config.PPO_CONFIG["train_batch_size_per_learner"],
                num_epochs=config.PPO_CONFIG["num_epochs"],
            )
        )

        self.algo = ppo_config.build_algo()
        print(f"PPO algorithm initialized with {self.num_workers} workers")