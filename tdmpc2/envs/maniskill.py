import gym  # for ManiSkill2 environment creation
import gymnasium as gym_new
import numpy as np
from tdmpc2.envs.wrappers.time_limit import TimeLimit

import mani_skill2.envs


MANISKILL_TASKS = {
    "lift-cube": dict(
        env="LiftCube-v0",
        control_mode="pd_ee_delta_pos",
    ),
    "pick-cube": dict(
        env="PickCube-v0",
        control_mode="pd_ee_delta_pos",
    ),
    "stack-cube": dict(
        env="StackCube-v0",
        control_mode="pd_ee_delta_pos",
    ),
    "pick-ycb": dict(
        env="PickSingleYCB-v0",
        control_mode="pd_ee_delta_pose",
    ),
    "turn-faucet": dict(
        env="TurnFaucet-v0",
        control_mode="pd_ee_delta_pose",
    ),
}


class GymToGymnasiumAdapter(gym_new.Env):
    """Adapter to make old gym environments compatible with gymnasium wrappers."""
    
    def __init__(self, gym_env):
        self._gym_env = gym_env
        self.action_space = gym_env.action_space
        self.observation_space = gym_env.observation_space
        
    def reset(self, seed=None, options=None):
        # Old gym reset returns only observation
        obs = self._gym_env.reset()
        return obs, {}
        
    def step(self, action):
        # Old gym step returns (obs, reward, done, info)
        obs, reward, done, info = self._gym_env.step(action)
        # New gym step should return (obs, reward, terminated, truncated, info)
        truncated = info.get("TimeLimit.truncated", False)
        terminated = done and not truncated
        return obs, reward, terminated, truncated, info
        
    def render(self, mode="human"):
        return self._gym_env.render(mode=mode)
        
    def close(self):
        return self._gym_env.close()
        
    @property
    def unwrapped(self):
        return self._gym_env.unwrapped


class ManiSkillWrapper(gym_new.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.observation_space = self.env.observation_space
        self.action_space = gym_new.spaces.Box(
            low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
            high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
            dtype=self.env.action_space.dtype,
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        reward, terminated, truncated = 0.0, False, False
        info = {}
        for _ in range(2):
            obs, r, term, trunc, info = self.env.step(action)
            reward += r
            terminated = terminated or term
            truncated = truncated or trunc
        return obs, reward, terminated, truncated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, args, **kwargs):
        return self.env.render(mode="cameras")


def make_env(cfg):
    """
    Make ManiSkill2 environment.
    """
    if cfg.task not in MANISKILL_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.obs == "state", "This task only supports state observations."
    task_cfg = MANISKILL_TASKS[cfg.task]
    env = gym.make(  # use old gym for ManiSkill2 env creation
        task_cfg["env"],
        obs_mode="state",
        control_mode=task_cfg["control_mode"],
        render_camera_cfgs=dict(width=384, height=384),
    )
    # Convert old gym env to gymnasium-compatible env
    env = GymToGymnasiumAdapter(env)
    env = ManiSkillWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env
