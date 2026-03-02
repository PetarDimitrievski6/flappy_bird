"""PPO trainer for FlappyBird."""
from ray.rllib.algorithms.ppo import PPOConfig
from .base_trainer import BaseTrainer
from environment import register_environment
import config


class PPOTrainer(BaseTrainer):
    def __init__(self, num_workers: int = config.DEFAULT_NUM_WORKERS):
        super().__init__("PPO")
        self.num_workers = num_workers
        register_environment()
        self.build_algo()

    def build_algo(self):
        num_gpus_per_learner, device_msg = self._setup_gpu()

        lr = config.PPO_CONFIG["lr"]
        if getattr(config, 'LR_SCHEDULE_ENABLED', False):
            end_factor = getattr(config, 'LR_SCHEDULE_END_FACTOR', 0.1)
            total_timesteps = config.DEFAULT_NUM_ITERATIONS * config.PPO_CONFIG["train_batch_size_per_learner"]
            lr = [
                (0, lr),
                (total_timesteps, lr * end_factor),
            ]

        ppo_config = (
            PPOConfig()
            .environment(env="flappy-bird")
            .env_runners(
                num_env_runners=self.num_workers,
                num_cpus_per_env_runner=1,
                num_envs_per_env_runner=config.PPO_CONFIG["num_envs_per_worker"],
                rollout_fragment_length=config.PPO_CONFIG["rollout_fragment_length"],
            )
            .learners(
                num_learners=0,
                num_gpus_per_learner=num_gpus_per_learner,
            )
            .training(
                lr=lr,
                gamma=config.PPO_CONFIG["gamma"],
                lambda_=config.PPO_CONFIG["lambda"],
                clip_param=config.PPO_CONFIG["clip_param"],
                vf_clip_param=config.PPO_CONFIG["vf_clip_param"],
                vf_loss_coeff=config.PPO_CONFIG["vf_loss_coeff"],
                entropy_coeff=config.PPO_CONFIG["entropy_coeff"],
                kl_coeff=config.PPO_CONFIG["kl_coeff"],
                kl_target=config.PPO_CONFIG["kl_target"],
                grad_clip=config.PPO_CONFIG["grad_clip"],
                train_batch_size_per_learner=config.PPO_CONFIG["train_batch_size_per_learner"],
                num_epochs=config.PPO_CONFIG["num_epochs"],
                minibatch_size=config.PPO_CONFIG["minibatch_size"],
                model=config.PPO_CONFIG["model"],
                vf_share_layers=config.PPO_CONFIG["model"]["vf_share_layers"],
            )
            .framework("torch")
        )

        self.algo = ppo_config.build_algo()
        print(f"PPO initialized with {self.num_workers} workers on {device_msg}")
