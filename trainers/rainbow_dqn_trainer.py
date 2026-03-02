"""Rainbow DQN trainer for FlappyBird."""
from ray.rllib.algorithms.dqn import DQNConfig
from .base_trainer import BaseTrainer
from environment import register_environment
import config


class RainbowDQNTrainer(BaseTrainer):
    def __init__(self, num_workers: int = 1):
        super().__init__("RainbowDQN")
        self.num_workers = num_workers
        register_environment()
        self.build_algo()

    def build_algo(self):
        num_gpus_per_learner, device_msg = self._setup_gpu()

        rainbow_config = (
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
            .training(
                lr=config.RAINBOW_DQN_CONFIG["lr"],
                gamma=config.RAINBOW_DQN_CONFIG["gamma"],
                train_batch_size_per_learner=config.RAINBOW_DQN_CONFIG["train_batch_size_per_learner"],
                train_batch_size=config.RAINBOW_DQN_CONFIG["train_batch_size_per_learner"],
                double_q=config.RAINBOW_DQN_CONFIG["double_q"],
                dueling=config.RAINBOW_DQN_CONFIG["dueling"],
                n_step=config.RAINBOW_DQN_CONFIG["n_step"],
                noisy=config.RAINBOW_DQN_CONFIG["noisy"],
                num_atoms=config.RAINBOW_DQN_CONFIG["num_atoms"],
                v_min=config.RAINBOW_DQN_CONFIG["v_min"],
                v_max=config.RAINBOW_DQN_CONFIG["v_max"],
                num_steps_sampled_before_learning_starts=config.RAINBOW_DQN_CONFIG["num_steps_sampled_before_learning_starts"],
                target_network_update_freq=config.RAINBOW_DQN_CONFIG["target_network_update_freq"],
                tau=config.RAINBOW_DQN_CONFIG["tau"],
                replay_buffer_config={
                    "type": "PrioritizedEpisodeReplayBuffer",
                    "capacity": config.RAINBOW_DQN_CONFIG["replay_buffer_capacity"],
                    "alpha": config.RAINBOW_DQN_CONFIG["replay_alpha"],
                    "beta": config.RAINBOW_DQN_CONFIG["replay_beta"],
                },
                model=config.RAINBOW_DQN_CONFIG["model"],
            )
            .framework("torch")
        )

        self.algo = rainbow_config.build_algo()

        components = []
        if config.RAINBOW_DQN_CONFIG["double_q"]:
            components.append("Double DQN")
        if config.RAINBOW_DQN_CONFIG["dueling"]:
            components.append("Dueling")
        components.append(f"Multi-step (n={config.RAINBOW_DQN_CONFIG['n_step']})")
        if config.RAINBOW_DQN_CONFIG["noisy"]:
            components.append("Noisy Nets")
        components.append(f"Distributional (C51, atoms={config.RAINBOW_DQN_CONFIG['num_atoms']})")
        components.append("Prioritized Replay")

        print(f"Rainbow DQN initialized with {self.num_workers} workers on {device_msg}")
        print(f"  Components: {', '.join(components)}")
        print(f"  Network: {config.RAINBOW_DQN_CONFIG['model']['fcnet_hiddens']}")
