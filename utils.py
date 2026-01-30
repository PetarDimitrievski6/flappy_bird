"""Utility functions for comparing algorithms."""
import matplotlib.pyplot as plt
import numpy as np


def compare_training_results(results_dict, save_path="comparison.png"):
    """
    Compare training results from different algorithms.

    Args:
        results_dict: Dictionary with algorithm names as keys and training histories as values
        save_path: Path to save the comparison plot
    """
    plt.figure(figsize=(12, 6))

    for algo_name, history in results_dict.items():
        rewards = [
            result.get('env_runners', {}).get('episode_return_mean', 0)
            for result in history
        ]
        plt.plot(rewards, label=algo_name, linewidth=2)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Mean Episode Reward', fontsize=12)
    plt.title('Algorithm Comparison on FlappyBird', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.show()


def print_comparison_summary(test_results_dict):
    """
    Print a comparison summary of test results.

    Args:
        test_results_dict: Dictionary with algorithm names as keys and test rewards as values
    """
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