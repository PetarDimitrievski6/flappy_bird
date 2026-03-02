"""CQL (Conservative Q-Learning) trainer for FlappyBird (discrete actions)."""
import copy
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.utils.annotations import override

from .base_trainer import BaseTrainer
from .cql_dqn_policy import CQLDQNTorchPolicy
from environment import register_environment
import config


class CQL(DQN):
    """DQN with a CQL-style loss for discrete action spaces."""

    @classmethod
    @override(DQN)
    def get_default_policy_class(cls, config: AlgorithmConfig):
        if config["framework"] == "torch":
            return CQLDQNTorchPolicy
        return super().get_default_policy_class(config)


class CQLDQNConfig(DQNConfig):
    """DQN config that restores replay buffer type to a string for old API."""

    def validate(self) -> None:
        if self.in_evaluation:
            rb_type = self.replay_buffer_config.get("type")
            if isinstance(rb_type, str):
                if "MultiAgentPrioritizedReplayBuffer" in rb_type:
                    from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import (
                        MultiAgentPrioritizedReplayBuffer,
                    )

                    self.replay_buffer_config["type"] = MultiAgentPrioritizedReplayBuffer
                elif "MultiAgentReplayBuffer" in rb_type:
                    from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
                        MultiAgentReplayBuffer,
                    )

                    self.replay_buffer_config["type"] = MultiAgentReplayBuffer
        super().validate()
        if not self.in_evaluation:
            rb_type = self.replay_buffer_config.get("type")
            if isinstance(rb_type, type):
                self.replay_buffer_config["type"] = rb_type.__name__


class CQLTrainer(BaseTrainer):
    """Trainer for CQL algorithm (DQN-based, old API stack)."""

    def __init__(self, num_workers: int = 1):
        super().__init__("CQL")
        self.num_workers = num_workers
        register_environment()
        self.build_algo()

    def build_algo(self):
        num_gpus, device_msg = self._setup_gpu()

        replay_buffer_config = copy.deepcopy(config.CQL_CONFIG["replay_buffer_config"])
        if not isinstance(replay_buffer_config.get("type"), str):
            replay_buffer_config["type"] = replay_buffer_config["type"].__name__

        cql_config = (
            CQLDQNConfig(algo_class=CQL)
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env="flappy-bird")
            .env_runners(
                num_env_runners=self.num_workers,
                num_cpus_per_env_runner=1,
                exploration_config=copy.deepcopy(config.CQL_CONFIG["exploration_config"]),
            )
            .resources(num_gpus=num_gpus)
            .training(
                lr=config.CQL_CONFIG["lr"],
                gamma=config.CQL_CONFIG["gamma"],
                train_batch_size=config.CQL_CONFIG["train_batch_size"],
                train_batch_size_per_learner=config.CQL_CONFIG["train_batch_size_per_learner"],
                num_steps_sampled_before_learning_starts=config.CQL_CONFIG["num_steps_sampled_before_learning_starts"],
                target_network_update_freq=config.CQL_CONFIG["target_network_update_freq"],
                replay_buffer_config=replay_buffer_config,
                double_q=config.CQL_CONFIG["double_q"],
                dueling=config.CQL_CONFIG["dueling"],
                n_step=config.CQL_CONFIG["n_step"],
            )
            .framework("torch")
        )
        cql_config.model = {**cql_config.model, **config.CQL_CONFIG["model"]}
        cql_config.cql_min_q_weight = config.CQL_CONFIG["cql_min_q_weight"]
        cql_config.cql_temperature = config.CQL_CONFIG["cql_temperature"]

        self.algo = cql_config.build_algo()
        print(f"CQL initialized with {self.num_workers} workers on {device_msg}")

    def get_action(self, observation):
        """Get action from the trained model (old API stack)."""
        if self.algo is None:
            raise ValueError("Algorithm not trained yet.")

        action = self.algo.compute_single_action(observation, explore=False)
        if isinstance(action, tuple):
            action = action[0]
        if hasattr(action, "item"):
            return action.item()
        return int(action)
