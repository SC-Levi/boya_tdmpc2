"""
Moore版本的BaseWorldModel

基于TDMPC2的WorldModel基类，为MooreWorldModel提供必要的基础结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from copy import deepcopy

from TDMPC2.algorithms.common import math, layers, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class BaseWorldModel(nn.Module):
    """
    修复版本的TD-MPC2 implicit world model架构。
    解决了action_dim为字符串时的类型问题。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            
            # 修复action_dim类型问题
            try:
                # 确保action_dim是整数
                action_dim = int(cfg.action_dim) if isinstance(cfg.action_dim, str) else cfg.action_dim
                
                self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), action_dim))
                for i in range(len(cfg.tasks)):
                    # 确保action_dims[i]也是整数
                    action_dims_i = int(cfg.action_dims[i]) if isinstance(cfg.action_dims[i], str) else cfg.action_dims[i]
                    self._action_masks[i, :action_dims_i] = 1.
            except Exception as e:
                print(f"警告: 创建action_masks失败: {e}")
                print("将继续初始化其他组件")
                
        self._encoder = layers.enc(cfg)
        self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
        self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
        self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
        self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
        self.init()

    def init(self):
        # Create params
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

        # Create modules
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)

        # Assign params to modules
        # We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    def __repr__(self):
        repr = 'TD-MPC2 World Model\n'
        modules = ['Encoder', 'Dynamics', 'Reward', 'Policy prior', 'Q-functions']
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._pi, self._Qs]):
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        
        在MOORE版本中，任务信息已经由任务编码器融合到潜在表示中，
        因此直接返回输入张量而不进行额外拼接。
        
        Args:
            x: 输入张量，通常是MOORE任务编码器的输出(v_c)
            task: 任务ID（不再使用）
            
        Returns:
            直接返回输入张量x
        """
        # 直接返回输入张量，不再拼接额外的任务嵌入
        return x

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """

        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """

        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """


        # Gaussian policy prior
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        if self.cfg.multitask: # Mask out unused action dimensions
            try:
                mean = mean * self._action_masks[task]
                log_std = log_std * self._action_masks[task]
                eps = eps * self._action_masks[task]
                action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
            except Exception as e:
                print(f"警告: 应用动作掩码失败: {e}")
                action_dims = None
        else: # No masking
            action_dims = None

        log_prob = math.gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1] if action_dims is None else action_dims
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict({
            "mean": mean,
            "log_std": log_std,
            "action_prob": 1.,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        })
        return action, info

    def Q(self, z, a, task, return_type='min', target=False, detach=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}



        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == 'all':
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2 