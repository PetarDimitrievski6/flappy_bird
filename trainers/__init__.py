"""Trainers module."""
import config
from .ppo_trainer import PPOTrainer
from .dqn_trainer import DQNTrainer
from .cql_trainer import CQLTrainer
from .rainbow_dqn_trainer import RainbowDQNTrainer
from .appo_trainer import APPOTrainer
from .impala_trainer import IMPALATrainer

__all__ = ['PPOTrainer', 'DQNTrainer', 'CQLTrainer', 'RainbowDQNTrainer', 'APPOTrainer', 'IMPALATrainer',
           'create_trainer']

TRAINER_MAP = {
    "PPO": lambda: PPOTrainer(num_workers=config.DEFAULT_NUM_WORKERS),
    "DQN": lambda: DQNTrainer(num_workers=1),
    "CQL": lambda: CQLTrainer(num_workers=1),
    "RAINBOW": lambda: RainbowDQNTrainer(num_workers=2),
    "RAINBOWDQN": lambda: RainbowDQNTrainer(num_workers=2),
    "APPO": lambda: APPOTrainer(num_workers=config.DEFAULT_NUM_WORKERS),
    "IMPALA": lambda: IMPALATrainer(num_workers=config.DEFAULT_NUM_WORKERS),
}


def create_trainer(algorithm: str):
    """Create a trainer instance for the given algorithm name."""
    key = algorithm.upper()
    factory = TRAINER_MAP.get(key)
    if factory is None:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                         f"Supported: {', '.join(k for k in TRAINER_MAP if k != 'RAINBOWDQN')}")
    return factory()
