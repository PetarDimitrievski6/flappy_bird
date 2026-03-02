"""Utility functions for comparing algorithms."""
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def smooth_rewards(rewards, window=10):
    """Apply moving average smoothing to rewards."""
    if len(rewards) < window:
        return rewards
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    return list(smoothed)


def extract_metrics(history):
    """Extract key metrics from training history."""
    metrics = {
        'rewards': [],
        'episode_lengths': [],
        'max_rewards': [],
        'min_rewards': [],
        'timesteps': [],
    }
    total_timesteps = 0

    for result in history:
        env_runners = result.get('env_runners') or {}
        if env_runners:
            metrics['rewards'].append(env_runners.get('episode_return_mean', 0))
            metrics['episode_lengths'].append(env_runners.get('episode_len_mean', 0))
            metrics['max_rewards'].append(env_runners.get('episode_return_max', 0))
            metrics['min_rewards'].append(env_runners.get('episode_return_min', 0))
        else:
            metrics['rewards'].append(result.get('episode_return_mean', 0))
            metrics['episode_lengths'].append(result.get('episode_len_mean', 0))
            metrics['max_rewards'].append(result.get('episode_return_max', 0))
            metrics['min_rewards'].append(result.get('episode_return_min', 0))

        total_timesteps += result.get('num_env_steps_sampled_this_iter', 0)
        metrics['timesteps'].append(total_timesteps)

    return metrics


def export_metrics_csv(history, algorithm_name, save_path=None):
    """Export training metrics to CSV file."""
    if save_path is None:
        save_path = f"{algorithm_name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    metrics = extract_metrics(history)

    with open(save_path, 'w') as f:
        f.write("iteration,reward_mean,reward_min,reward_max,episode_length,timesteps\n")
        for i in range(len(metrics['rewards'])):
            f.write(f"{i+1},{metrics['rewards'][i]},{metrics['min_rewards'][i]},"
                    f"{metrics['max_rewards'][i]},{metrics['episode_lengths'][i]},"
                    f"{metrics['timesteps'][i]}\n")

    print(f"Metrics exported to: {save_path}")
    return save_path


def plot_training_curve(history, algorithm_name, save_path=None, show=True, smoothing=10):
    """Plot detailed training curve with smoothing."""
    metrics = extract_metrics(history)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reward plot
    ax1 = axes[0, 0]
    ax1.fill_between(range(len(metrics['rewards'])),
                     metrics['min_rewards'], metrics['max_rewards'],
                     alpha=0.3, label='Min-Max Range')
    ax1.plot(metrics['rewards'], alpha=0.3, label='Raw')
    if smoothing > 0:
        smoothed = smooth_rewards(metrics['rewards'], smoothing)
        ax1.plot(range(smoothing-1, len(metrics['rewards'])), smoothed,
                linewidth=2, label=f'Smoothed (window={smoothing})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title(f'{algorithm_name} - Training Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Episode length plot
    ax2 = axes[0, 1]
    ax2.plot(metrics['episode_lengths'], alpha=0.5)
    if smoothing > 0:
        smoothed_len = smooth_rewards(metrics['episode_lengths'], smoothing)
        ax2.plot(range(smoothing-1, len(metrics['episode_lengths'])), smoothed_len, linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Episode Length')
    ax2.set_title(f'{algorithm_name} - Episode Length')
    ax2.grid(True, alpha=0.3)

    # Max reward progression
    ax3 = axes[1, 0]
    ax3.plot(metrics['max_rewards'], alpha=0.5)
    cummax = np.maximum.accumulate([r if not np.isnan(r) else 0 for r in metrics['max_rewards']])
    ax3.plot(cummax, linewidth=2, label='Cumulative Max')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Max Reward')
    ax3.set_title(f'{algorithm_name} - Max Reward Progression')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Timesteps vs reward
    ax4 = axes[1, 1]
    valid_idx = [i for i, r in enumerate(metrics['rewards']) if not np.isnan(r)]
    ax4.scatter([metrics['timesteps'][i] for i in valid_idx],
                [metrics['rewards'][i] for i in valid_idx], alpha=0.5, s=10)
    ax4.set_xlabel('Total Timesteps')
    ax4.set_ylabel('Mean Reward')
    ax4.set_title(f'{algorithm_name} - Sample Efficiency')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = f"{algorithm_name}_training_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curve saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def compare_training_results(results_dict, save_path="comparison.png", smoothing=10):
    """Compare training results from different algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reward comparison
    ax1 = axes[0]
    for algo_name, history in results_dict.items():
        rewards = [
            result.get('env_runners', {}).get('episode_return_mean', 0)
            for result in history
        ]
        ax1.plot(rewards, alpha=0.3)
        if smoothing > 0 and len(rewards) >= smoothing:
            smoothed = smooth_rewards(rewards, smoothing)
            ax1.plot(range(smoothing-1, len(rewards)), smoothed,
                    linewidth=2, label=algo_name)
        else:
            ax1.plot(rewards, linewidth=2, label=algo_name)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('Algorithm Comparison - Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Episode length comparison
    ax2 = axes[1]
    for algo_name, history in results_dict.items():
        lengths = [
            result.get('env_runners', {}).get('episode_len_mean', 0)
            for result in history
        ]
        if smoothing > 0 and len(lengths) >= smoothing:
            smoothed = smooth_rewards(lengths, smoothing)
            ax2.plot(range(smoothing-1, len(lengths)), smoothed,
                    linewidth=2, label=algo_name)
        else:
            ax2.plot(lengths, linewidth=2, label=algo_name)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean Episode Length')
    ax2.set_title('Algorithm Comparison - Episode Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.show()


def print_comparison_summary(test_results_dict):
    """Print a comparison summary of test results."""
    print("\n" + "=" * 70)
    print(" ALGORITHM COMPARISON SUMMARY")
    print("=" * 70)

    for algo_name, rewards in test_results_dict.items():
        avg = np.mean(rewards)
        std = np.std(rewards)
        max_r = np.max(rewards)
        min_r = np.min(rewards)

        print(f"\n{algo_name}:")
        print(f"  Average: {avg:6.2f} ± {std:5.2f}")
        print(f"  Max:     {max_r:6.2f}")
        print(f"  Min:     {min_r:6.2f}")

    print("=" * 70)


def compute_training_stats(history):
    """Compute comprehensive training statistics."""
    metrics = extract_metrics(history)

    valid_rewards = [r for r in metrics['rewards'] if not np.isnan(r)]

    stats = {
        'total_iterations': len(history),
        'total_timesteps': metrics['timesteps'][-1] if metrics['timesteps'] else 0,
        'final_reward': valid_rewards[-1] if valid_rewards else 0,
        'best_reward': max(valid_rewards) if valid_rewards else 0,
        'mean_reward': np.mean(valid_rewards) if valid_rewards else 0,
        'std_reward': np.std(valid_rewards) if valid_rewards else 0,
        'best_max_reward': max([r for r in metrics['max_rewards'] if not np.isnan(r)]) if metrics['max_rewards'] else 0,
        'mean_episode_length': np.mean([l for l in metrics['episode_lengths'] if not np.isnan(l)]) if metrics['episode_lengths'] else 0,
    }

    return stats
