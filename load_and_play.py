"""Load and play a trained Flappy Bird model - PROPER METHOD."""
import os
import sys
import glob
from pathlib import Path
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.impala import IMPALA
from ray.rllib.algorithms.appo import APPO
from trainers.cql_trainer import CQL as CQLAlgo
import numpy as np
import torch
from environment import register_environment, make_flappy_env
import config


def _is_checkpoint_dir(path: Path) -> bool:
    """Check if path is a valid RLlib checkpoint directory."""
    if (path / "rllib_checkpoint.json").is_file():
        return True
    if list(path.glob("algorithm_state.*")):
        return True
    if list(path.glob("checkpoint-*")):
        return True
    return False


def _is_nested_checkpoint_component(path: Path) -> bool:
    """Check if path is a nested component of a checkpoint (env_runner, learner_group, etc.)."""
    component_names = {"env_runner", "learner_group", "learner", "env_to_module_connector", "module_to_env_connector"}
    return path.name in component_names


def _find_model_checkpoints(models_dir: Path, algorithm: str) -> list:
    """Find valid checkpoint directories for the given algorithm."""
    if not models_dir.exists():
        return []

    search_algorithms = [algorithm]
    if algorithm.upper() in ["RAINBOW", "RAINBOWDQN"]:
        search_algorithms = ["RainbowDQN", "DQN"]

    checkpoints = []
    for algo_name in search_algorithms:
        algo_dir = models_dir / algo_name
        if not algo_dir.exists():
            continue
        for candidate in algo_dir.rglob("*"):
            if candidate.is_dir() and _is_checkpoint_dir(candidate):
                # Skip nested checkpoint components
                if _is_nested_checkpoint_component(candidate):
                    continue
                # Skip if any parent is a checkpoint (this is a nested dir)
                is_nested = False
                for parent in candidate.parents:
                    if parent == algo_dir:
                        break
                    if _is_checkpoint_dir(parent) and not _is_nested_checkpoint_component(parent):
                        is_nested = True
                        break
                if not is_nested:
                    checkpoints.append(str(candidate))
    return checkpoints


def find_checkpoints(algorithm="PPO", limit=5):
    """Find available checkpoints."""
    models_dir = Path(__file__).resolve().parent / "models"
    checkpoints = _find_model_checkpoints(models_dir, algorithm)
    checkpoints = sorted(set(checkpoints), key=os.path.getmtime, reverse=True)
    if checkpoints:
        return checkpoints[:limit]

    ray_results_dir = os.path.expanduser("~/ray_results")
    if not os.path.exists(ray_results_dir):
        print(f"No checkpoints found. Directory doesn't exist: {ray_results_dir}")
        return []

    search_algorithms = [algorithm]
    if algorithm.upper() in ["RAINBOW", "RAINBOWDQN"]:
        search_algorithms = ["RainbowDQN", "DQN"]

    training_dirs = []
    for algo_name in search_algorithms:
        pattern = f"{ray_results_dir}/{algo_name}_flappy-bird_*"
        training_dirs.extend(glob.glob(pattern))

    training_dirs = sorted(set(training_dirs), key=os.path.getmtime, reverse=True)
    for training_dir in training_dirs[:limit]:
        for root, dirs, files in os.walk(training_dir):
            for d in sorted(dirs, reverse=True):
                if d.startswith("checkpoint_"):
                    candidate = Path(root) / d
                    if _is_checkpoint_dir(candidate):
                        checkpoints.append(str(candidate))
                    break
            if checkpoints and checkpoints[-1].startswith(training_dir):
                break

    return checkpoints


def play_with_model(checkpoint_path, num_episodes=10, render=True):
    """Load a checkpoint and play episodes."""
    print("="*70)
    print(" LOADING TRAINED MODEL")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}\n")

    # Register environment
    register_environment()

    # Determine algorithm from checkpoint path
    if "APPO" in checkpoint_path:
        AlgoClass = APPO
        algo_name = "APPO"
    elif "PPO" in checkpoint_path:
        AlgoClass = PPO
        algo_name = "PPO"
    elif "CQL" in checkpoint_path:
        AlgoClass = CQLAlgo
        algo_name = "CQL"
    elif "RainbowDQN" in checkpoint_path or "Rainbow" in checkpoint_path:
        AlgoClass = DQN  # Rainbow is a DQN variant
        algo_name = "RainbowDQN"
    elif "IMPALA" in checkpoint_path:
        AlgoClass = IMPALA
        algo_name = "IMPALA"
    elif "DQN" in checkpoint_path:
        AlgoClass = DQN
        algo_name = "DQN"
    else:
        print("Could not determine algorithm from checkpoint path")
        return

    try:
        # Load the algorithm from checkpoint
        print(f"Loading {algo_name} model...")
        algo = AlgoClass.from_checkpoint(checkpoint_path)
        print(f"{algo_name} model loaded successfully!\n")

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a trained model (run: python main.py)")
        print("2. Check the checkpoint path exists")
        print("3. Try a different checkpoint")
        return

    # Create environment for playing
    render_mode = "human" if render else None
    env = make_flappy_env(
        {
            "use_lidar": False,
            "render_mode": render_mode,
            "top_touch_penalty": config.TOP_TOUCH_PENALTY,
        }
    )

    print("="*70)
    print(f" PLAYING {num_episodes} EPISODES")
    print("="*70)

    episode_rewards = []

    def _compute_action(observation):
        rl_module = None
        try:
            rl_module = algo.get_module()
        except Exception:
            rl_module = None

        if rl_module is not None and hasattr(rl_module, "forward_inference"):
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
            action_output = rl_module.forward_inference({"obs": obs_tensor})

            if "actions" in action_output:
                action = action_output["actions"][0]
                if torch.is_tensor(action):
                    return int(action.item())
                if hasattr(action, "item"):
                    return int(action.item())
                return int(action)
            if "action_dist_inputs" in action_output:
                action_logits = action_output["action_dist_inputs"][0]
                if not torch.is_tensor(action_logits):
                    action_logits = torch.from_numpy(np.asarray(action_logits))
                return int(torch.argmax(action_logits).item())
            if "q_values" in action_output:
                q_values = action_output["q_values"][0]
                if not torch.is_tensor(q_values):
                    q_values = torch.from_numpy(np.asarray(q_values))
                return int(torch.argmax(q_values).item())

        # Fallback for older API stacks
        action = algo.compute_single_action(observation, explore=False)
        if isinstance(action, tuple):
            action = action[0]
        if torch.is_tensor(action):
            return int(action.item())
        if hasattr(action, "item"):
            return int(action.item())
        return int(action)

    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done:
                # Get action from trained model
                action = _compute_action(obs)

                # Take action
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {steps}")

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        env.close()
        algo.stop()

    # Print summary
    if episode_rewards:
        print("\n" + "="*70)
        print(" RESULTS")
        print("="*70)
        print(f"Average Reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
        print(f"Best Reward: {max(episode_rewards):.2f}")
        print(f"Worst Reward: {min(episode_rewards):.2f}")
        print("="*70)


def main():
    """Main function."""
    print("="*70)
    print(" FLAPPY BIRD - LOAD AND PLAY TRAINED MODEL")
    print("="*70)

    algorithm = "PPO"
    num_episodes = 5

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1].upper() in ["PPO", "DQN", "CQL", "RAINBOW", "RAINBOWDQN", "IMPALA", "APPO"]:
            algorithm = sys.argv[1].upper()
            if algorithm in ["RAINBOW", "RAINBOWDQN"]:
                algorithm = "RainbowDQN"
        elif os.path.exists(sys.argv[1]):
            # Direct checkpoint path provided
            checkpoint_path = sys.argv[1]
            if len(sys.argv) > 2:
                num_episodes = int(sys.argv[2])
            play_with_model(checkpoint_path, num_episodes, render=True)
            return

    if len(sys.argv) > 2:
        num_episodes = int(sys.argv[2])

    # Find available checkpoints
    print(f"\nSearching for {algorithm} checkpoints...")
    checkpoints = find_checkpoints(algorithm, limit=10)

    if not checkpoints:
        print(f"\nNo {algorithm} checkpoints found!")
        print("\nTo train a model, run: python main.py")
        print("Results will be saved to: ./models/")
        return

    # Display checkpoints
    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    print("="*70)
    for i, cp in enumerate(checkpoints, 1):
        basename = os.path.basename(cp)
        timestamp = basename.split(f'{algorithm}_flappy-bird_')[-1] if algorithm in basename else basename
        print(f"{i}. {timestamp}")

    # Use the latest (first) checkpoint
    checkpoint_path = checkpoints[0]
    print(f"\nUsing latest checkpoint: {os.path.basename(checkpoint_path)}")

    # Play with the model
    play_with_model(checkpoint_path, num_episodes, render=True)


if __name__ == "__main__":
    print("\nUsage:")
    print("  python load_and_play.py                        # Play latest PPO model (5 episodes)")
    print("  python load_and_play.py PPO 10                 # Play latest PPO model (10 episodes)")
    print("  python load_and_play.py CQL 10                 # Play latest CQL model (10 episodes)")
    print("  python load_and_play.py IMPALA 10              # Play latest IMPALA model (10 episodes)")
    print("  python load_and_play.py DQN                    # Play latest DQN model")
    print("  python load_and_play.py /path/to/checkpoint 10 # Play specific checkpoint")
    print("\n")

    main()
