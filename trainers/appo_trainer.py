"""APPO trainer for FlappyBird."""
from ray.rllib.algorithms.appo import APPOConfig
from .base_trainer import BaseTrainer
from environment import register_environment
import config


class APPOTrainer(BaseTrainer):
    def __init__(self, num_workers: int = config.DEFAULT_NUM_WORKERS):
        super().__init__("APPO")
        self.num_workers = num_workers
        register_environment()
        self.build_algo()

    def build_algo(self):
        num_gpus_per_learner, device_msg = self._setup_gpu()

        appo_config = (
            APPOConfig()
            .environment(env="flappy-bird")
            .env_runners(
                num_env_runners=self.num_workers,
                num_cpus_per_env_runner=1,
                num_envs_per_env_runner=config.APPO_CONFIG["num_envs_per_worker"],
                rollout_fragment_length=config.APPO_CONFIG["rollout_fragment_length"],
            )
            .learners(
                num_learners=0,
                num_gpus_per_learner=num_gpus_per_learner,
            )
            .training(
                lr=config.APPO_CONFIG["lr"],
                gamma=config.APPO_CONFIG["gamma"],
                lambda_=config.APPO_CONFIG["lambda"],
                train_batch_size_per_learner=config.APPO_CONFIG["train_batch_size_per_learner"],
                num_epochs=config.APPO_CONFIG["num_epochs"],
                minibatch_size=config.APPO_CONFIG["minibatch_size"],
                vf_loss_coeff=config.APPO_CONFIG["vf_loss_coeff"],
                entropy_coeff=config.APPO_CONFIG["entropy_coeff"],
                clip_param=config.APPO_CONFIG["clip_param"],
                grad_clip=config.APPO_CONFIG["grad_clip"],
                vtrace=config.APPO_CONFIG["vtrace"],
                model=config.APPO_CONFIG["model"],
            )
            .framework("torch")
        )

        self.algo = appo_config.build_algo()
        print(f"APPO initialized with {self.num_workers} workers on {device_msg}")
