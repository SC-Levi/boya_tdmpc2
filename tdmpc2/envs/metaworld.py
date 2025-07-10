import numpy as np
import gymnasium as gym
from .wrappers.time_limit import TimeLimit

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaWorldToGymnasiumAdapter(gym.Env):
    """Adapter to make MetaWorld environments compatible with gymnasium wrappers."""

    def __init__(self, metaworld_env):
        super().__init__()
        self._metaworld_env = metaworld_env
        self.action_space = metaworld_env.action_space
        self.observation_space = metaworld_env.observation_space

    def reset(self, seed=None, options=None):
        # MetaWorld reset returns only observation
        obs = self._metaworld_env.reset()
        return obs, {}

    def step(self, action):
        # MetaWorld step returns (obs, reward, done, info)
        obs, reward, done, info = self._metaworld_env.step(action)
        # New gymnasium step should return (obs, reward, terminated, truncated, info)
        truncated = info.get("TimeLimit.truncated", False)
        terminated = done and not truncated
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self._metaworld_env.render(*args, **kwargs)

    def close(self):
        return self._metaworld_env.close()

    @property
    def unwrapped(self):
        return self._metaworld_env
        
    def __getattr__(self, name):
        # Forward any missing attributes to the underlying MetaWorld environment
        return getattr(self._metaworld_env, name)


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env._freeze_rand_vec = False

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = obs.astype(np.float32)
        self.env.step(np.zeros(self.env.action_space.shape))
        return obs, info

    def step(self, action):
        reward = 0
        terminated, truncated = False, False
        info = {}
        for _ in range(2):
            obs, r, term, trunc, info = self.env.step(action.copy())
            reward += r
            terminated = terminated or term
            truncated = truncated or trunc
        obs = obs.astype(np.float32)
        return obs, reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render(
            offscreen=True, resolution=(384, 384), camera_name=self.camera_name
        ).copy()


def make_env(cfg):
    """
    Make Meta-World environment.
    """
    env_id = cfg.task.split("-", 1)[-1] + "-v2-goal-observable"
    if (
        not cfg.task.startswith("mw-")
        or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    ):
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.obs == "state", "This task only supports state observations."
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=cfg.seed)
    # Convert MetaWorld env to gymnasium-compatible env
    env = MetaWorldToGymnasiumAdapter(env)
    env = MetaWorldWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env
