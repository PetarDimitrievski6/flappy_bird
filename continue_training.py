"""Continue training a saved agent from a checkpoint."""
import sys
from pathlib import Path
from trainers import create_trainer
import config


def find_latest_checkpoint(algorithm: str) -> str | None:
    """Find the latest checkpoint for the given algorithm."""
    models_dir = Path(__file__).parent / "models"

    algo_names = [algorithm.upper()]
    if algorithm.upper() in ["RAINBOW", "RAINBOWDQN"]:
        algo_names = ["RainbowDQN", "Rainbow"]

    for algo_name in algo_names:
        algo_dir = models_dir / algo_name
        if not algo_dir.exists():
            continue

        checkpoints = []
        for candidate in algo_dir.iterdir():
            if candidate.is_dir():
                if (candidate / "rllib_checkpoint.json").exists():
                    checkpoints.append(candidate)
                elif list(candidate.glob("algorithm_state.*")):
                    checkpoints.append(candidate)

        if checkpoints:
            checkpoints.sort(key=lambda p: p.name, reverse=True)
            return str(checkpoints[0])

    return None


def continue_training(
    algorithm: str = "PPO",
    checkpoint_path: str = None,
    additional_iterations: int = 1000,
    test_after: bool = True,
):
    """Continue training from a saved checkpoint."""
    print(f"\n{'='*70}")
    print(f" CONTINUE TRAINING {algorithm.upper()}")
    print(f"{'='*70}")

    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(algorithm)
        if checkpoint_path is None:
            print(f"No checkpoint found for {algorithm}. Train a model first: python main.py")
            return None

    print(f"Loading checkpoint: {checkpoint_path}")

    trainer = create_trainer(algorithm)
    trainer.load(checkpoint_path)
    print(f"Successfully loaded checkpoint!")

    print(f"\nContinuing training for {additional_iterations} more iterations...")
    trainer.train(num_iterations=additional_iterations)

    new_checkpoint = trainer.save()
    print(f"\nNew checkpoint saved to: {new_checkpoint}")

    if test_after:
        trainer.test(
            num_episodes=5,
            render=True,
            log_action_distribution=config.DEFAULT_LOG_ACTION_DISTRIBUTION,
        )

    trainer.cleanup()
    return new_checkpoint


def print_usage():
    print("""
Usage:
  python continue_training.py <algorithm> [iterations] [checkpoint_path]

Examples:
  python continue_training.py PPO                    # Continue latest PPO for 1000 iterations
  python continue_training.py PPO 500                # Continue latest PPO for 500 iterations
  python continue_training.py DQN 2000               # Continue latest DQN for 2000 iterations
  python continue_training.py PPO 1000 /path/to/checkpoint  # Continue specific checkpoint

Supported algorithms: PPO, DQN, CQL, Rainbow, APPO, IMPALA
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()

        print("\nAvailable checkpoints:")
        models_dir = Path(__file__).parent / "models"
        if models_dir.exists():
            for algo_dir in models_dir.iterdir():
                if algo_dir.is_dir():
                    checkpoints = list(algo_dir.glob("*/rllib_checkpoint.json"))
                    if checkpoints:
                        print(f"  {algo_dir.name}:")
                        for cp in checkpoints:
                            print(f"    - {cp.parent.name}")
        sys.exit(0)

    algorithm = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    checkpoint = sys.argv[3] if len(sys.argv) > 3 else None

    continue_training(
        algorithm=algorithm,
        checkpoint_path=checkpoint,
        additional_iterations=iterations,
        test_after=True,
    )
