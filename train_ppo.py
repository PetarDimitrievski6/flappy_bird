"""Train PPO on FlappyBird."""
from main import train_single_algorithm


def main():
    train_single_algorithm(algorithm="PPO")


if __name__ == "__main__":
    main()
