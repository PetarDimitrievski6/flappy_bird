"""Base trainer class for all RL algorithms."""
import torch
import numpy as np
import gymnasium


class BaseTrainer:
    """Base class for training RL algorithms on FlappyBird."""

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.algo = None
        self.training_history = []

    def train(self, num_iterations: int = 100):
        """Train the agent."""
        if self.algo is None:
            raise ValueError("Algorithm not initialized. Call build_algo() first.")

        print(f"\nStarting {self.algorithm_name} training for {num_iterations} iterations...")
        print("-" * 60)

        for i in range(num_iterations):
            result = self.algo.train()
            self.training_history.append(result)
            reward = result.get('env_runners', {}).get('episode_return_mean', 'N/A')
            print(f"Iteration {i + 1}/{num_iterations}: Mean Reward = {reward:.2f}")

        print("-" * 60)
        print("Training complete!")
        return self.training_history

    def get_action(self, observation: np.ndarray) -> int:
        """Get action from the trained model."""
        if self.algo is None:
            raise ValueError("Algorithm not trained yet.")

        rl_module = self.algo.get_module()
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
        action_output = rl_module.forward_inference({"obs": obs_tensor})

        # Handle different output formats for different algorithms
        if "action_dist_inputs" in action_output:
            # PPO and other policy gradient methods
            action_logits = action_output["action_dist_inputs"][0]
            return torch.argmax(action_logits).item()
        elif "actions" in action_output:
            # DQN and other value-based methods
            action = action_output["actions"][0]
            if torch.is_tensor(action):
                return action.item()
            return int(action)
        else:
            # Try Q-values directly
            if "q_values" in action_output:
                q_values = action_output["q_values"][0]
                return torch.argmax(q_values).item()
            else:
                raise KeyError(f"Unknown action output format. Available keys: {action_output.keys()}")

    def test(self, num_episodes: int = 5, render: bool = True):
        """Test the trained agent."""
        if self.algo is None:
            raise ValueError("Algorithm not trained yet.")

        import flappy_bird_gymnasium

        render_mode = "human" if render else None
        episode_rewards = []

        print(f"\nTesting {self.algorithm_name} agent for {num_episodes} episodes...")
        print("-" * 60)

        for episode in range(num_episodes):
            env = gymnasium.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
            obs, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.get_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
            env.close()

        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print("-" * 60)
        print(f"Average reward: {avg_reward:.2f}")
        print("-" * 60)

        return episode_rewards

    def save(self, path: str = None) -> str:
        """Save the trained model."""
        if self.algo is None:
            raise ValueError("Algorithm not trained yet.")

        checkpoint_dir = self.algo.save(path)
        print(f"Model saved to: {checkpoint_dir}")
        return str(checkpoint_dir)

    def load(self, checkpoint_dir: str):
        """Load a trained model."""
        if self.algo is None:
            raise ValueError("Algorithm not initialized.")

        self.algo.restore(checkpoint_dir)
        print(f"Model loaded from: {checkpoint_dir}")

    def cleanup(self):
        """Clean up resources."""
        if self.algo is not None:
            self.algo.stop()
            self.algo = None
            print(f"{self.algorithm_name} resources cleaned up.")