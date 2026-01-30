import gymnasium
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
import numpy as np
import torch


def make_flappy_env(config):
    """Factory function that imports and creates the environment"""
    import flappy_bird_gymnasium
    return gymnasium.make("FlappyBird-v0", use_lidar=False)


if __name__ == '__main__':
    # Register the environment with RLlib
    tune.register_env("flappy-bird", make_flappy_env)

    # Create PPO config
    config = (
        PPOConfig()
        .environment(env="flappy-bird")
        .env_runners(num_env_runners=2)
        .training(
            lr=0.0002,
            train_batch_size_per_learner=2000,
            num_epochs=10,
        )
    )

    ppo = config.build_algo()

    # Train
    for i in range(4):
        result = ppo.train()
        print(f"Iteration {i}: Reward = {result.get('env_runners', {}).get('episode_return_mean', 'N/A')}")

    # Test the trained agent using the NEW API
    import flappy_bird_gymnasium

    env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)

    # Get the trained RLModule
    rl_module = ppo.get_module()

    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Convert observation to tensor with batch dimension
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

        # Use the new API for inference
        action_output = rl_module.forward_inference({"obs": obs_tensor})

        # The output key might be different - try to find the right one
        if "action_dist_inputs" in action_output:
            # Get action from distribution inputs
            action_logits = action_output["action_dist_inputs"][0]
            action = torch.argmax(action_logits).item()
        elif "actions" in action_output:
            action = action_output["actions"][0]
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            if hasattr(action, 'item'):
                action = action.item()
        else:
            # Print available keys to debug
            print(f"Available keys: {action_output.keys()}")
            raise KeyError("Could not find action in output")

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Total reward: {total_reward}")
    env.close()
    ppo.stop()