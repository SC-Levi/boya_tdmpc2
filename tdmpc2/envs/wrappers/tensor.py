from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
    """
    Wrapper for converting numpy arrays to torch tensors.
    """

    def __init__(self, env):
        super().__init__(env)

    @property
    def max_episode_steps(self):
        return self.env.max_episode_steps

    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def _try_f32_tensor(self, x):
        x = torch.from_numpy(x)
        if x.dtype == torch.float64:
            x = x.float()
        return x

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def reset(self, task_idx=None):
        obs, info = self.env.reset()
        return self._obs_to_tensor(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action.numpy())
        info = defaultdict(float, info)
        info["success"] = float(info["success"])
        return (
            self._obs_to_tensor(obs),
            torch.tensor(reward, dtype=torch.float32),
            done,
            truncated,
            info,
        )
