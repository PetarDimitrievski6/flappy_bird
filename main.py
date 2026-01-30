"""Main script for training and comparing RL algorithms on FlappyBird."""
from trainers import PPOTrainer, DQNTrainer
from utils import compare_training_results, print_comparison_summary
import config


def train_single_algorithm(algorithm="PPO"):
    """Train a single algorithm."""
    print(f"\n{'='*70}")
    print(f" TRAINING {algorithm.upper()} ON FLAPPYBIRD")
    print(f"{'='*70}")

    # Select trainer
    if algorithm.upper() == "PPO":
        trainer = PPOTrainer(num_workers=config.DEFAULT_NUM_WORKERS)
    elif algorithm.upper() == "DQN":
        trainer = DQNTrainer(num_workers=1)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Train with same number of iterations
    trainer.train(num_iterations=config.DEFAULT_NUM_ITERATIONS)

    # Save model
    checkpoint = trainer.save()

    # Test with same number of episodes
    trainer.test(num_episodes=2, render=True)

    # Cleanup
    trainer.cleanup()

    return checkpoint


def compare_algorithms():
    """Train and compare multiple algorithms."""
    print(f"\n{'='*70}")
    print(" COMPARING ALGORITHMS ON FLAPPYBIRD")
    print(f"{'='*70}")

    algorithms = ["PPO", "DQN"]
    trainers = {}
    training_histories = {}
    test_results = {}

    # Train each algorithm
    for algo_name in algorithms:
        print(f"\n\n{'='*70}")
        print(f" TRAINING {algo_name}")
        print(f"{'='*70}")

        if algo_name == "PPO":
            trainer = PPOTrainer(num_workers=config.DEFAULT_NUM_WORKERS)
        elif algo_name == "DQN":
            trainer = DQNTrainer(num_workers=1)

        # Train with same iterations for fair comparison
        history = trainer.train(num_iterations=config.DEFAULT_NUM_ITERATIONS)
        training_histories[algo_name] = history

        # Save
        checkpoint = trainer.save()
        print(f"{algo_name} checkpoint: {checkpoint}")

        # Test without rendering (faster) with same number of episodes
        rewards = trainer.test(num_episodes=config.DEFAULT_TEST_EPISODES, render=False)
        test_results[algo_name] = rewards

        trainers[algo_name] = trainer

    # Compare results
    compare_training_results(training_histories, save_path="algorithm_comparison.png")
    print_comparison_summary(test_results)

    # Cleanup all trainers
    for trainer in trainers.values():
        trainer.cleanup()


if __name__ == '__main__':
    # Choose mode:

    # Mode 1: Train a single algorithm
    train_single_algorithm(algorithm="PPO")
    # train_single_algorithm(algorithm="DQN")

    # Mode 2: Compare multiple algorithms (both trained for 200 iterations)
    # compare_algorithms()