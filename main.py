"""Main script for training and comparing RL algorithms on FlappyBird."""
from trainers import create_trainer
from utils import compare_training_results, print_comparison_summary
import config


def train_single_algorithm(algorithm="PPO", num_iterations=None):
    """Train a single algorithm."""
    if num_iterations is None:
        num_iterations = config.DEFAULT_NUM_ITERATIONS

    print(f"\n{'='*70}")
    print(f" TRAINING {algorithm.upper()} ON FLAPPYBIRD")
    print(f"{'='*70}")

    trainer = create_trainer(algorithm)

    trainer.train(num_iterations=num_iterations)

    stats = trainer.get_stats()
    if stats:
        print(f"\nTraining Statistics:")
        print(f"  Best Reward: {stats['best_reward']:.2f}")
        print(f"  Best Max Reward: {stats['best_max_reward']:.2f}")
        print(f"  Final Reward: {stats['final_reward']:.2f}")
        print(f"  Total Timesteps: {stats['total_timesteps']:,}")

    trainer.export_metrics()
    trainer.plot_training_curve(show=False)

    checkpoint = trainer.save()

    trainer.test(
        num_episodes=2,
        render=True,
        log_action_distribution=config.DEFAULT_LOG_ACTION_DISTRIBUTION,
    )

    trainer.cleanup()
    return checkpoint


def compare_algorithms():
    """Train and compare multiple algorithms."""
    print(f"\n{'='*70}")
    print(" COMPARING ALGORITHMS ON FLAPPYBIRD")
    print(f"{'='*70}")

    algorithms = ["PPO", "DQN", "CQL", "Rainbow", "APPO", "IMPALA"]
    trainers = {}
    training_histories = {}
    test_results = {}

    for algo_name in algorithms:
        print(f"\n\n{'='*70}")
        print(f" TRAINING {algo_name}")
        print(f"{'='*70}")

        trainer = create_trainer(algo_name)
        history = trainer.train(num_iterations=config.DEFAULT_NUM_ITERATIONS)
        training_histories[algo_name] = history

        checkpoint = trainer.save()
        print(f"{algo_name} checkpoint: {checkpoint}")

        rewards = trainer.test(
            num_episodes=config.DEFAULT_TEST_EPISODES,
            render=False,
            log_action_distribution=config.DEFAULT_LOG_ACTION_DISTRIBUTION,
        )
        test_results[algo_name] = rewards
        trainers[algo_name] = trainer

    compare_training_results(training_histories, save_path="algorithm_comparison.png")
    print_comparison_summary(test_results)

    for trainer in trainers.values():
        trainer.cleanup()


if __name__ == '__main__':
    train_single_algorithm(algorithm="PPO")
    # compare_algorithms()
