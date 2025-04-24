from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from TDMPC2.algorithms.common import layers, math, init
from TDMPC2.algorithms.common.world_model import WorldModel
from TDMPC2.algorithms.common.moore_layers import MooreTaskEmbeddingModule
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class MooreWorldModel(WorldModel):
    """
    集成Moore混合专家架构的TD-MPC2世界模型
    扩展原始的WorldModel，将任务嵌入替换为Moore风格的混合专家机制
    
    Args:
        cfg: 配置对象，除标准WorldModel配置外，还支持以下Moore特有配置：
            - n_experts: 专家数量，默认4
            - temperature: 专家权重的softmax温度，默认1.0
            - use_softmax: 是否使用softmax归一化专家权重，默认True
            - expert_hidden_dims: 专家网络隐藏层维度，默认使用[cfg.mlp_dim, cfg.mlp_dim]
            - debug_task_emb: 是否启用任务嵌入调试输出，默认False
    """

    def __init__(self, cfg):
        # 初始化父类，但我们将覆盖任务嵌入部分
        super(WorldModel, self).__init__()  # 直接调用nn.Module的初始化
        self.cfg = cfg
        
        # 提取Moore专家架构相关配置
        self.n_experts = getattr(cfg, 'n_experts', 4)
        self.temperature = getattr(cfg, 'temperature', 1.0)
        self.use_softmax = getattr(cfg, 'use_softmax', True)
        self.expert_hidden_dims = getattr(cfg, 'expert_hidden_dims', [cfg.mlp_dim, cfg.mlp_dim])
        self.debug_task_emb = getattr(cfg, 'debug_task_emb', False)
        
        # 初始化任务嵌入缓存
        self._cached_features = None
        
        if cfg.multitask:
            # 使用Moore混合专家架构
            self._moore_task_emb = MooreTaskEmbeddingModule(
                n_tasks=len(cfg.tasks),
                input_dim=cfg.obs_dim,  # 使用观测维度作为输入维度
                hidden_dims=self.expert_hidden_dims,  # 专家网络隐藏层维度
                embedding_dim=cfg.task_dim,  # 输出维度与原task_dim相同
                n_experts=self.n_experts,
                max_norm=1,  # 保持与原始嵌入相同的norm约束
                temperature=self.temperature,
                use_softmax=self.use_softmax
            )
            
            # 注册action_masks缓冲区
            self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
            for i in range(len(cfg.tasks)):
                self._action_masks[i, :cfg.action_dims[i]] = 1.
        
        # 初始化其他组件与原WorldModel相同
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

    def get_expert_stats(self):
        """
        获取任务-专家分配的统计信息
        
        Returns:
            专家使用统计信息，包括平均权重和任务计数
        """
        if hasattr(self, '_moore_task_emb') and self.cfg.multitask:
            return self._moore_task_emb.get_expert_stats()
        return None

    def task_emb(self, x, task):
        """
        使用Moore混合专家架构生成任务嵌入
        
        Args:
            x: 输入特征 [batch_size, feature_dim]
            task: 任务ID [batch_size] 或整数
            
        Returns:
            拼接了任务嵌入的特征 [batch_size, feature_dim + task_dim]
        """
        if not self.cfg.multitask:
            return x
            
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
            
        # 使用Moore混合专家架构计算任务嵌入
        # 根据是否有已缓存的特征向量决定使用哪个输入
        if self._cached_features is not None:
            # 使用已缓存的特征向量
            emb = self._moore_task_emb(self._cached_features, task)
        else:
            # 使用当前输入特征
            emb = self._moore_task_emb(x, task)
            
        # 调试信息：打印任务嵌入的形状和范数
        if self.debug_task_emb and not torch.jit.is_scripting():
            print(f"Task: {task.cpu().detach().numpy()}, Emb shape: {emb.shape}")
            norm = torch.norm(emb, dim=1, keepdim=True)
            print(f"Emb norm: {norm.cpu().detach().numpy().ravel()}")
            print(f"Emb values[0:5]: {emb[0, :5].cpu().detach().numpy()}")
            
            # 获取专家权重
            if hasattr(self._moore_task_emb, 'task_encoder'):
                weights = self._moore_task_emb.task_encoder(task)
                print(f"Expert weights: {weights.cpu().detach().numpy()}")
            
        # 与原始task_emb保持相同的维度处理
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
            
        return torch.cat([x, emb], dim=-1)
    
    def encode(self, obs, task):
        """
        重写encode方法，在编码前缓存原始观测
        
        Args:
            obs: 观测 [batch_size, obs_dim]
            task: 任务ID [batch_size] 或整数
            
        Returns:
            编码后的潜在状态 [batch_size, latent_dim]
        """
        # 缓存原始观测，用于后续的任务嵌入计算
        if self.cfg.multitask:
            # 只保存原始形状的观测，用于计算任务嵌入
            if obs.ndim <= 2:
                self._cached_features = obs
        
        # 编码过程，与原WorldModel保持一致
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            encoded = torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        else:
            encoded = self._encoder[self.cfg.obs](obs)
            
        # 清除缓存
        self._cached_features = None
            
        return encoded
        
    def __repr__(self):
        """自定义打印方法，包含Moore专有信息"""
        repr = 'TD-MPC2 Moore World Model\n'
        if self.cfg.multitask:
            repr += f"Moore config: experts={self.n_experts}, temperature={self.temperature}, softmax={self.use_softmax}\n"
        modules = ['Encoder', 'Dynamics', 'Reward', 'Policy prior', 'Q-functions']
        for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._pi, self._Qs]):
            repr += f"{modules[i]}: {m}\n"
        repr += "Learnable parameters: {:,}".format(self.total_params)
        return repr 