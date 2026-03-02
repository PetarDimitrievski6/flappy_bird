"""Environment setup for FlappyBird."""
import gymnasium
from gymnasium.wrappers import TimeLimit
from ray import tune


class TopTouchPenaltyWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, penalty: float):
        super().__init__(env)
        self._top_touch_penalty = float(penalty)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        top_touch = False
        try:
            player_y = getattr(self.env.unwrapped, "_player_y", None)
            if player_y is not None and player_y < 0:
                top_touch = True
        except Exception:
            top_touch = False

        if top_touch and not (terminated or truncated):
            reward = self._top_touch_penalty if reward is None else min(float(reward), self._top_touch_penalty)
            info = info or {}
            info["top_touch_penalty_applied"] = True

        return obs, reward, terminated, truncated, info


class MasteryBonusWrapper(gymnasium.Wrapper):
    """Give huge bonus reward when agent reaches max episode steps (mastery)."""
    def __init__(self, env: gymnasium.Env, bonus: float, max_steps: int):
        super().__init__(env)
        self._bonus = float(bonus)
        self._max_steps = max_steps
        self._current_steps = 0

    def reset(self, **kwargs):
        self._current_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_steps += 1

        # If truncated due to time limit (not terminated by death), give bonus
        if truncated and not terminated and self._current_steps >= self._max_steps:
            reward = (reward or 0) + self._bonus
            info = info or {}
            info["mastery_bonus_applied"] = True
            info["mastery_bonus"] = self._bonus

        return obs, reward, terminated, truncated, info


def make_flappy_env(config):
    import flappy_bird_gymnasium

    env_config = config if isinstance(config, dict) else {}
    use_lidar = bool(env_config.get("use_lidar", False))
    render_mode = env_config.get("render_mode")

    env = gymnasium.make("FlappyBird-v0", use_lidar=use_lidar, render_mode=render_mode)

    # Apply top touch penalty wrapper
    penalty = env_config.get("top_touch_penalty")
    if penalty is None:
        try:
            import config as project_config
            penalty = getattr(project_config, "TOP_TOUCH_PENALTY", None)
        except Exception:
            penalty = None

    if penalty is not None:
        env = TopTouchPenaltyWrapper(env, penalty=penalty)

    # Apply max episode steps limit (prevents infinite episodes)
    max_steps = env_config.get("max_episode_steps")
    if max_steps is None:
        try:
            import config as project_config
            max_steps = getattr(project_config, "MAX_EPISODE_STEPS", None)
        except Exception:
            max_steps = None

    # Get mastery bonus
    mastery_bonus = env_config.get("max_steps_bonus_reward")
    if mastery_bonus is None:
        try:
            import config as project_config
            mastery_bonus = getattr(project_config, "MAX_STEPS_BONUS_REWARD", None)
        except Exception:
            mastery_bonus = None

    if max_steps is not None and max_steps > 0:
        env = TimeLimit(env, max_episode_steps=max_steps)
        if mastery_bonus is not None and mastery_bonus > 0:
            env = MasteryBonusWrapper(env, bonus=mastery_bonus, max_steps=max_steps)

    return env


def register_environment():
    tune.register_env("flappy-bird", make_flappy_env)
