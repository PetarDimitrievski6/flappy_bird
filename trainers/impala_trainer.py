"""IMPALA trainer for FlappyBird."""
from ray.rllib.algorithms.impala import IMPALAConfig
from .base_trainer import BaseTrainer
from environment import register_environment
import config


class IMPALATrainer(BaseTrainer):
    def __init__(self, num_workers: int = config.DEFAULT_NUM_WORKERS):
        super().__init__("IMPALA")
        self.num_workers = num_workers
        register_environment()
        self.build_algo()

    def build_algo(self):
        num_gpus_per_learner, device_msg = self._setup_gpu()

        impala_config = (
            IMPALAConfig()
            .environment(env="flappy-bird")
            .env_runners(
                num_env_runners=self.num_workers,
                num_cpus_per_env_runner=1,
                num_envs_per_env_runner=config.IMPALA_CONFIG["num_envs_per_worker"],
                rollout_fragment_length=config.IMPALA_CONFIG["rollout_fragment_length"],
            )
            .learners(
                num_learners=0,
                num_gpus_per_learner=num_gpus_per_learner,
            )
            .training(
                lr=config.IMPALA_CONFIG["lr"],
                gamma=config.IMPALA_CONFIG["gamma"],
                train_batch_size_per_learner=config.IMPALA_CONFIG["train_batch_size_per_learner"],
                train_batch_size=config.IMPALA_CONFIG["train_batch_size_per_learner"],
                minibatch_size=config.IMPALA_CONFIG["minibatch_size"],
                num_epochs=config.IMPALA_CONFIG["num_epochs"],
                vf_loss_coeff=config.IMPALA_CONFIG["vf_loss_coeff"],
                entropy_coeff=config.IMPALA_CONFIG["entropy_coeff"],
                vtrace=config.IMPALA_CONFIG["vtrace"],
                vtrace_clip_rho_threshold=config.IMPALA_CONFIG["vtrace_clip_rho_threshold"],
                vtrace_clip_pg_rho_threshold=config.IMPALA_CONFIG["vtrace_clip_pg_rho_threshold"],
                grad_clip=config.IMPALA_CONFIG["grad_clip"],
                learner_queue_size=config.IMPALA_CONFIG["learner_queue_size"],
                num_gpu_loader_threads=config.IMPALA_CONFIG["num_gpu_loader_threads"],
                broadcast_interval=config.IMPALA_CONFIG["broadcast_interval"],
                model=config.IMPALA_CONFIG["model"],
            )
            .reporting(min_time_s_per_iteration=config.IMPALA_CONFIG["min_time_s_per_iteration"])
            .framework("torch")
        )

        self.algo = impala_config.build_algo()
        print(f"IMPALA initialized with {self.num_workers} workers on {device_msg}")
