"""Base trainer class for all RL algorithms."""
import math
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import config
from environment import make_flappy_env


class BaseTrainer:
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.algo = None
        self.training_history = []

    @staticmethod
    def _setup_gpu():
        """Configure GPU optimizations. Returns (num_gpus_per_learner, device_msg)."""
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            return 1, f"GPU ({torch.cuda.get_device_name(0)})"
        return 0, "CPU"

    def train(self, num_iterations: int = 100, checkpoint_freq: int = 100,
              early_stopping: bool = None, early_stopping_threshold: float = None, early_stopping_patience: int = None):
        if self.algo is None:
            raise ValueError("Algorithm not initialized. Call build_algo() first.")

        if early_stopping is None:
            early_stopping = getattr(config, 'EARLY_STOPPING_ENABLED', False)
        if early_stopping_threshold is None:
            early_stopping_threshold = getattr(config, 'EARLY_STOPPING_REWARD_THRESHOLD', 500.0)
        if early_stopping_patience is None:
            early_stopping_patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 50)

        print(f"\nStarting {self.algorithm_name} training for {num_iterations} iterations...")
        if early_stopping:
            print(f"Early stopping: threshold={early_stopping_threshold}, patience={early_stopping_patience}")
        print("-" * 60)

        best_mean_reward = float('-inf')
        no_improvement_count = 0
        total_timesteps = 0
        start_time = datetime.now()

        for i in range(num_iterations):
            result = self.algo.train()
            self.training_history.append(result)

            env_runners = result.get('env_runners') or {}
            if env_runners:
                mean_reward = env_runners.get('episode_return_mean', 0)
                min_reward = env_runners.get('episode_return_min', 0)
                max_reward = env_runners.get('episode_return_max', 0)
                episode_len_mean = env_runners.get('episode_len_mean', 0)
            else:
                mean_reward = result.get('episode_return_mean', result.get('episode_reward_mean', 0))
                min_reward = result.get('episode_return_min', result.get('episode_reward_min', 0))
                max_reward = result.get('episode_return_max', result.get('episode_reward_max', 0))
                episode_len_mean = result.get('episode_len_mean', 0)

            timesteps_this_iter = result.get('num_env_steps_sampled_this_iter',
                                             result.get('timesteps_this_iter', 0))
            total_timesteps += timesteps_this_iter

            iter_num = i + 1

            is_nan_reward = mean_reward is None or (isinstance(mean_reward, float) and math.isnan(mean_reward))

            if is_nan_reward:
                print(f"Iter {iter_num:3d}/{num_iterations}: "
                      f"Reward:    nan (no episodes completed this iteration) | "
                      f"Ep Len:   nan")
            else:
                print(f"Iter {iter_num:3d}/{num_iterations}: "
                      f"Reward: {mean_reward:6.2f} (min: {min_reward:6.2f}, max: {max_reward:6.2f}) | "
                      f"Ep Len: {episode_len_mean:5.1f}")
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            # Periodic checkpoint
            if checkpoint_freq > 0 and iter_num % checkpoint_freq == 0:
                periodic_path = self._save_periodic_checkpoint(iter_num)
                print(f"  → Checkpoint saved: {periodic_path}")

            # Early stopping checks
            if early_stopping and not is_nan_reward:
                if mean_reward >= early_stopping_threshold:
                    print(f"\n🎯 Early stopping: Reached reward threshold ({mean_reward:.2f} >= {early_stopping_threshold})")
                    break
                if no_improvement_count >= early_stopping_patience:
                    print(f"\n⏹ Early stopping: No improvement for {early_stopping_patience} iterations")
                    break

        elapsed = datetime.now() - start_time
        print("-" * 60)
        print(f"Training complete! ({elapsed.total_seconds():.1f}s, {total_timesteps:,} timesteps)")
        return self.training_history


    def _save_periodic_checkpoint(self, iteration: int) -> str:
        base_dir = Path(__file__).resolve().parents[1] / "models" / self.algorithm_name
        base_dir.mkdir(parents=True, exist_ok=True)
        path = str(base_dir / f"iter_{iteration}")
        self.algo.save(path)
        return path

    def get_action(self, observation: np.ndarray) -> int:
        if self.algo is None:
            raise ValueError("Algorithm not trained yet.")

        try:
            action = self.algo.compute_single_action(observation, explore=False)
            if isinstance(action, tuple):
                action = action[0]
            if torch.is_tensor(action):
                return action.item()
            return int(action)
        except Exception:
            pass

        rl_module = None
        try:
            rl_module = self.algo.get_module()
        except Exception:
            rl_module = None

        if rl_module is None or not hasattr(rl_module, "forward_inference"):
            action = self.algo.compute_single_action(observation, explore=False)
            if isinstance(action, tuple):
                action = action[0]
            if torch.is_tensor(action):
                return action.item()
            return int(action)

        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
        action_output = rl_module.forward_inference({"obs": obs_tensor})

        if "actions" in action_output:
            action = action_output["actions"][0]
            if torch.is_tensor(action):
                return action.item()
            return int(action)
        elif "action_dist_inputs" in action_output:
            action_logits = action_output["action_dist_inputs"][0]
            if not torch.is_tensor(action_logits):
                action_logits = torch.from_numpy(np.asarray(action_logits))
            return int(torch.argmax(action_logits).item())
        elif "q_values" in action_output:
            q_values = action_output["q_values"][0]
            if not torch.is_tensor(q_values):
                q_values = torch.from_numpy(np.asarray(q_values))
            return int(torch.argmax(q_values).item())
        else:
            raise KeyError(f"Unknown action output format. Keys: {action_output.keys()}")

    def test(self, num_episodes: int = 5, render: bool = True, log_action_distribution: bool = False):
        if self.algo is None:
            raise ValueError("Algorithm not trained yet.")

        render_mode = "human" if render else None
        episode_rewards = []
        action_counts = None
        action_space_n = None

        print(f"\nTesting {self.algorithm_name} agent for {num_episodes} episodes...")
        print("-" * 60)

        for episode in range(num_episodes):
            env = make_flappy_env({
                "use_lidar": False,
                "render_mode": render_mode,
                "top_touch_penalty": config.TOP_TOUCH_PENALTY,
            })
            obs, _ = env.reset()
            total_reward = 0
            done = False

            if log_action_distribution and action_counts is None:
                if hasattr(env.action_space, "n"):
                    action_space_n = env.action_space.n
                    action_counts = np.zeros(action_space_n, dtype=np.int64)

            while not done:
                action = self.get_action(obs)
                if action_counts is not None and 0 <= int(action) < action_space_n:
                    action_counts[int(action)] += 1
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")
            env.close()

        avg_reward = sum(episode_rewards) / len(episode_rewards)
        print("-" * 60)
        print(f"Average reward: {avg_reward:.2f}")
        if action_counts is not None:
            total_actions = int(action_counts.sum())
            if total_actions > 0:
                dist = [f"{i}:{c} ({c/total_actions*100:.1f}%)" for i, c in enumerate(action_counts)]
                print(f"Action distribution: {', '.join(dist)}")
        print("-" * 60)
        return episode_rewards

    def save(self, path: str = None) -> str:
        if self.algo is None:
            raise ValueError("Algorithm not trained yet.")

        if path is None:
            base_dir = Path(__file__).resolve().parents[1] / "models" / self.algorithm_name
            base_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(base_dir / timestamp)

        checkpoint_result = self.algo.save(path)
        checkpoint_dir = path
        if hasattr(checkpoint_result, "checkpoint"):
            checkpoint = checkpoint_result.checkpoint
            if checkpoint is not None:
                checkpoint_dir = getattr(checkpoint, "path", None) or checkpoint.to_directory()
        elif isinstance(checkpoint_result, str):
            checkpoint_dir = checkpoint_result

        print(f"Model saved to: {checkpoint_dir}")
        return str(checkpoint_dir)

    def load(self, checkpoint_dir: str):
        if self.algo is None:
            raise ValueError("Algorithm not initialized.")
        self.algo.restore(checkpoint_dir)
        print(f"Model loaded from: {checkpoint_dir}")

    def export_metrics(self, save_path=None):
        """Export training metrics to CSV."""
        from utils import export_metrics_csv
        if not self.training_history:
            print("No training history to export.")
            return None
        return export_metrics_csv(self.training_history, self.algorithm_name, save_path)

    def plot_training_curve(self, save_path=None, show=True, smoothing=10):
        """Plot training curves."""
        from utils import plot_training_curve
        if not self.training_history:
            print("No training history to plot.")
            return None
        return plot_training_curve(self.training_history, self.algorithm_name, save_path, show, smoothing)

    def get_stats(self):
        """Get training statistics."""
        from utils import compute_training_stats
        if not self.training_history:
            return {}
        return compute_training_stats(self.training_history)

    def cleanup(self):
        if self.algo is not None:
            self.algo.stop()
            self.algo = None
            print(f"{self.algorithm_name} resources cleaned up.")
