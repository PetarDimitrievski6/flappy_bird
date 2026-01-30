"""Environment setup for FlappyBird."""
import gymnasium
from ray import tune


def make_flappy_env(config):
    """Factory function that creates a FlappyBird environment."""
    import flappy_bird_gymnasium
    return gymnasium.make("FlappyBird-v0", use_lidar=False)


def register_environment():
    """Register the FlappyBird environment with RLlib."""
    tune.register_env("flappy-bird", make_flappy_env)