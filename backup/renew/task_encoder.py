import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

from expert_module import MooreExpertsModule


class MooreTaskEncoder(nn.Module):
    """
    Moore架构的任务编码器
    
    使用混合专家方法生成任务特定的潜在表示，包含：
    1. 输入复制到多个专家 (InputLayer)
    2. 通过共享网络并行处理 (ParallelLayer)
    3. 正交化专家输出 (OrthogonalLayer1D)
    4. 门控机制聚合专家输出
    """
    
    def __init__(self, obs_dim, latent_dim, hidden_dim=256, n_experts=4, temperature=1.0, use_softmax=True, n_contexts=10, use_checkpoint=False):
        """
        初始化MooreTaskEncoder
        
        Args:
            obs_dim: 观察空间维度
            latent_dim: 潜在表示维度
            hidden_dim: 隐藏层维度
            n_experts: 专家数量
            temperature: 门控softmax温度参数
            use_softmax: 是否使用softmax进行门控权重归一化
            n_contexts: 任务上下文数量
            use_checkpoint: 是否使用梯度检查点
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.temperature = temperature
        self.use_softmax = use_softmax
        self.cfg = type('Config', (), {'n_contexts': n_contexts})
        self.use_checkpoint = use_checkpoint
        
        # 使用已经优化过的 MooreExpertsModule
        self.experts = MooreExpertsModule(
            input_dim=obs_dim,
            hidden_dims=[hidden_dim],
            output_dim=latent_dim,
            n_experts=n_experts
        )
        
        # 基于任务ID的门控网络
        self.gate = nn.Linear(n_contexts, n_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight, gain=nn.init.calculate_gain('linear'))

    def _experts_forward(self, flat_obs):
        """独立的专家前向传播函数，用于梯度检查点"""
        # 使用优化过的 MooreExpertsModule
        return self.experts(flat_obs)

    def _gate_forward(self, c_onehot):
        """独立的门控前向传播函数，用于梯度检查点"""
        logits = self.gate(c_onehot)
        if self.use_softmax:
            w = F.softmax(logits / self.temperature, dim=-1)
        else:
            raw = F.relu(logits / self.temperature)
            w = raw / (raw.sum(-1, keepdim=True) + 1e-8)
        return w

    def forward(self, obs, c=None):
        """
        标准化接口:
        obs: [B, obs_dim] 或 [B, T, obs_dim]li
        c: None或[B]格式的任务ID
        """
        # 1. 处理任务ID - 直接标准化为[B]格式
        if c is None:
            c_ids = torch.zeros(obs.shape[0] if obs.dim() == 2 else obs.shape[0] * obs.shape[1], 
                               dtype=torch.long, device=obs.device)
        elif isinstance(c, int):
            c_ids = torch.full((obs.shape[0] if obs.dim() == 2 else obs.shape[0] * obs.shape[1],), 
                               c, device=obs.device, dtype=torch.long)
        else:
            # 确保c是张量并展平
            c = c.to(obs.device).long()
            if obs.dim() > 2 and c.dim() == 1:
                # 如果obs是[B,T,D]而c是[B]，扩展c
                c_ids = c.repeat_interleave(obs.shape[1])
            else:
                c_ids = c.reshape(-1)  # 直接展平任何其他格式
        
        # 2. 安全检查任务ID范围
        c_ids = torch.clamp(c_ids, 0, self.cfg.n_contexts - 1)
        
        # 3. 扁平化观测
        if obs.dim() > 2:
            B, T, D = obs.shape
            flat_obs = obs.reshape(B * T, D)
        else:
            flat_obs = obs
        
        # 4. 计算one-hot并获取门控权重
        c_onehot = F.one_hot(c_ids, num_classes=self.cfg.n_contexts).float().to(self.gate.weight.device)
        w = F.softmax(self.gate(c_onehot) / self.temperature, dim=-1).to(obs.device)
        
        # 5. 获取专家特征并加权聚合
        expert_feats = self.experts(flat_obs).permute(1, 0, 2)  # [N, E, D]
        v_flat = (w.unsqueeze(-1) * expert_feats).sum(dim=1)    # [N, D]
        
        # 6. 恢复原始形状
        if obs.dim() > 2:
            return v_flat.view(B, T, self.latent_dim)
        return v_flat

    def get_task_representation(self, obs, task_ids=None):
        """获取任务表示的便捷方法"""
        task_repr = self.forward(obs, task_ids)
        return task_repr
        
    def __repr__(self):
        """自定义打印方法"""
        return (f"MooreTaskEncoder(obs_dim={self.obs_dim}, "
                f"latent_dim={self.latent_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"n_experts={self.n_experts}, "
                f"temperature={self.temperature}, "
                f"use_softmax={self.use_softmax})") 