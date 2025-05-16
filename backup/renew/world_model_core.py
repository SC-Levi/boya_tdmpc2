import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
from moore_tdmpc.expert_module import MooreExpertsModule
from typing import Optional 

class MoETransitionRewardModel(nn.Module):
    """
    共享专家网络的 Dynamics + Reward 模块
    """
    def __init__(self,
                 latent_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 n_experts: int = 4,
                 temperature: float = 1.0,
                 use_softmax: bool = True,
                 reward_dim: int = 101,
                 use_mixed_precision: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.temperature = temperature
        self.use_softmax = use_softmax
        self.reward_dim = reward_dim
        self.use_mixed_precision = use_mixed_precision
        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))

        # 1) 共享 MoE 专家网络
        self.experts = MooreExpertsModule(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=latent_dim,
            n_experts=n_experts
        )

        # 2) 预投影层
        self.pre_proj = nn.Linear(latent_dim + action_dim, hidden_dim)

        # 3) 门控网络 - 只为动态预测输出权重
        self._gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts)  # 只输出一个头的权重
        )

        # 4) 使用MLP作为reward_head，增强表达能力
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),  # 或 F.mish 如果没有直接的nn.Mish
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, reward_dim)
        )
        
        # 5) 动态head只保留一个
        self.head_W = nn.Parameter(torch.empty(latent_dim, latent_dim))
        self.head_b = nn.Parameter(torch.zeros(latent_dim))
        nn.init.xavier_uniform_(self.head_W)

    def dyn_head(self, z_feat, *_, **__):
        """
        只用融合后的 dyn_feat，动态从 head_W/head_b 切片，
        确保跟着 model.to(device) 一起搬。
        """
        W = self.head_W[0, :, :self.latent_dim]   # [D, D], 已在正确 device
        b = self.head_b[0, :self.latent_dim]      # [D]
        return z_feat @ W + b                     # [N, D]

    def forward(self, z: torch.Tensor, a: torch.Tensor, task_ids=None):
        """
        z:[N,D]  a:[N,A] → next_z, reward_logits
        """
        N = z.size(0)
        h = self.pre_proj(torch.cat((z, a), dim=-1))              # [N,H]

        # 获取动态和奖励专家的权重
        logits_dyn = self._gate(h)                                # [N,E]
        temp = max(0.5, self.temperature * 0.999 ** self.training_step.item())
        w_dyn = F.softmax(logits_dyn / temp, dim=-1)
        
        # 对奖励使用相同或单独的权重(可以是相同的专家权重)
        w_rew = w_dyn  # 共享权重简化版

        expert_feats = self.experts(h).permute(1, 0, 2)           # [N,E,D]
        dyn_feat = torch.einsum('bed,be->bd', expert_feats, w_dyn)
        rew_feat = torch.einsum('bed,be->bd', expert_feats, w_rew)  # [N,D]
        
        # 动态预测
        SOFT_CLAMP = 0.3
        delta_raw = dyn_feat @ self.head_W + self.head_b
        delta_z = SOFT_CLAMP * torch.tanh(delta_raw)
        next_z = z + delta_z
        
        # 奖励预测 - 使用MLP处理依赖于(z,a)的特征
        reward_logits = self.reward_head(rew_feat)  # [N, reward_dim]
        
        with torch.no_grad(): self.training_step += 1
        
        return next_z, reward_logits

    def __repr__(self):
        """自定义打印方法"""
        return (f"MoETransitionRewardModel(latent_dim={self.latent_dim}, "
                f"action_dim={self.action_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"n_experts={self.n_experts}, "
                f"temperature={self.temperature}, "
                f"use_softmax={self.use_softmax}, "
                f"reward_dim={self.reward_dim}, "
                f"fused_head=True, "
                f"use_mixed_precision={self.use_mixed_precision})")
                
    def set_debug_mode(self, debug_mode=True):
        """设置调试模式"""
        self.debug_mode = getattr(self, 'debug_mode', False)
        self.debug_mode = debug_mode
        return self
        

 