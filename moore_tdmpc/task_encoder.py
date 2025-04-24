import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

from moore_tdmpc.layers import InputLayer, ParallelLayer, OrthogonalLayer1D, MooreExpertsModule
from moore_tdmpc.utils.profiler import profiled


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
    
    @profiled
    def forward(self, obs, c=None):
        """
        obs: [B, obs_dim] or [T, B, obs_dim]
        c:   int or [B] or [B, T] or [T, B] 任务 ID，如果为None则默认为任务ID 0
        return v_c: [B, latent_dim] or [T, B, latent_dim]
        """
        # 初始化形状信息
        orig_shape = obs.shape
        
        # 确保c不为None，默认为0
        if c is None:
            c = 0
        
        # --- 1) 处理 c_ids 匹配 obs 扁平化后的一一对应关系 ---
        if isinstance(c, int):
            if obs.dim() > 2:
                H, B, _ = obs.shape
                c_ids = torch.full((H * B,), c, device=obs.device, dtype=torch.long)
            else:
                c_ids = torch.full((obs.shape[0],), c, device=obs.device, dtype=torch.long)
        else:
            # 确保c是LongTensor
            c = c.to(obs.device).long()
            
            if obs.dim() > 2:
                H, B, _ = obs.shape
                
                # 检查c是否已经是3D张量 [T, B, ?]
                if c.dim() == 3:
                    c_ids = c.view(-1)  # 直接展平
                # 检查c是否是匹配的2D张量 [T, B] 或 [B, T]
                elif c.dim() == 2:
                    if c.shape[0] == H and c.shape[1] == B:
                        # 如果已经是 [T, B] 形状，直接展平
                        c_ids = c.reshape(-1)
                    elif c.shape[0] == B and c.shape[1] == H:
                        # 如果是 [B, T] 形状，需要转置后展平
                        c_ids = c.transpose(0, 1).reshape(-1)
                    else:
                        error_msg = f"Context shape {c.shape} doesn't match obs shape {obs.shape}"
                        raise ValueError(error_msg)
                else:
                    # 1D张量 [B]，需要扩展到每个时间步以匹配扁平化的obs
                    try:
                        c_ids = c.unsqueeze(0).expand(H, -1).reshape(-1)
                        # 检查长度是否匹配
                        expected_len = H * B
                        if c_ids.shape[0] != expected_len:
                            # 修复长度
                            if c.shape[0] == B:
                                c_ids = c.repeat(H)
                    except Exception as e:
                        # 处理异常情况
                        c_ids = torch.zeros(H * B, dtype=torch.long, device=obs.device)
                        raise
            else:
                # 普通2D输入 [B, obs_dim]
                if c.dim() > 1:
                    c_ids = c.reshape(-1)  # 展平任何多维度
                else:
                    c_ids = c
        
        # 确保索引范围在有效范围内
        if c_ids.max() >= self.cfg.n_contexts:
            c_ids = torch.clamp(c_ids, 0, self.cfg.n_contexts - 1)
        
        # one-hot gating
        try:
            c_onehot = F.one_hot(c_ids, num_classes=self.cfg.n_contexts).float()
            
            # 确保张量在同一设备上
            c_onehot = c_onehot.to(self.gate.weight.device)
            
            if self.training and self.use_checkpoint:
                w = checkpoint(self._gate_forward, c_onehot)
            else:
                w = self._gate_forward(c_onehot)
            
            # 确保w在正确的设备上
            w = w.to(obs.device)
        except Exception as e:
            raise
        
        # --- 2) 扁平化 obs 并使用优化后的MooreExpertsModule --- 
        try:
            # 扁平化输入
            if obs.dim() > 2:
                H, B, D = orig_shape
                flat_obs = obs.reshape(H * B, D)
            else:
                flat_obs = obs
            
            # 使用优化后的MooreExpertsModule
            if self.training and self.use_checkpoint:
                expert_out = checkpoint(self._experts_forward, flat_obs)
            else:
                expert_out = self.experts(flat_obs)
            
            # expert_out: [n_experts, N, latent_dim]
            # 转成 [N, n_experts, latent_dim] 便于与 w [N, n_experts] 进行加权聚合
            expert_feats = expert_out.permute(1, 0, 2).contiguous()  # [N, E, latent_dim]
            
            # --- 3) 加权聚合 & 恢复原始形状 ---
            v_flat = (w.unsqueeze(-1) * expert_feats).sum(dim=1)  # [N, latent_dim]
            v_c = v_flat.view(*orig_shape[:-1], self.latent_dim)
            
            return v_c
        except Exception as e:
            raise

    @profiled
    def get_task_representation(self, obs, task_ids=None):
        """获取任务表示的便捷方法"""
        task_repr = self.forward(obs, task_ids)
        return task_repr
        
    def get_expert_weights(self, obs, task_ids=None):
        """获取专家权重的便捷方法"""
        # 如果没有任务ID，均匀分配专家权重
        if task_ids is None:
            if obs.dim() > 2:
                H, B, _ = obs.shape
                weights = torch.ones(H * B, self.n_experts, device=obs.device) / self.n_experts
                return weights.view(H, B, self.n_experts)
            else:
                batch_size = obs.shape[0]
                return torch.ones(batch_size, self.n_experts, device=obs.device) / self.n_experts
        
        # 处理任务ID并生成one-hot编码
        if isinstance(task_ids, int):
            if obs.dim() > 2:
                H, B, _ = obs.shape
                c_ids = torch.full((H * B,), task_ids, device=obs.device, dtype=torch.long)
            else:
                c_ids = torch.full((obs.shape[0],), task_ids, device=obs.device, dtype=torch.long)
        else:
            # 确保task_ids是LongTensor
            task_ids = task_ids.to(obs.device).long()
            
            if obs.dim() > 2:
                H, B, _ = obs.shape
                
                # 检查task_ids是否已经是3D张量 [T, B, ?]
                if task_ids.dim() == 3:
                    c_ids = task_ids.view(-1)  # 直接展平
                # 检查task_ids是否是匹配的2D张量 [T, B] 或 [B, T]
                elif task_ids.dim() == 2:
                    if task_ids.shape[0] == H and task_ids.shape[1] == B:
                        # 如果已经是 [T, B] 形状，直接展平
                        c_ids = task_ids.reshape(-1)
                    elif task_ids.shape[0] == B and task_ids.shape[1] == H:
                        # 如果是 [B, T] 形状，需要转置后展平
                        c_ids = task_ids.transpose(0, 1).reshape(-1)
                    else:
                        raise ValueError(f"Context shape {task_ids.shape} doesn't match obs shape {obs.shape}")
                else:
                    # 1D张量 [B]，需要扩展到每个时间步以匹配扁平化的obs
                    try:
                        c_ids = task_ids.unsqueeze(0).expand(H, -1).reshape(-1)
                        # Add extra check to verify c_ids length matches flattened obs
                        expected_len = H * B
                        if c_ids.shape[0] != expected_len:
                            # Try to fix
                            if task_ids.shape[0] == B:
                                c_ids = task_ids.repeat(H)
                    except Exception as e:
                        print(f"[ERROR] Failed to expand task_ids: {str(e)}")
                        # Handle exceptional case by creating a default array
                        c_ids = torch.zeros(H * B, dtype=torch.long, device=obs.device)
                        raise
            else:
                # 普通2D输入 [B, obs_dim]
                if task_ids.dim() > 1:
                    c_ids = task_ids.reshape(-1)  # 展平任何多维度
                else:
                    c_ids = task_ids
            
        # 确保索引范围在n_contexts内
        if c_ids.max() >= self.cfg.n_contexts:
            c_ids = torch.clamp(c_ids, 0, self.cfg.n_contexts - 1)
        
        # Create one-hot encoding
        c_onehot = F.one_hot(c_ids, num_classes=self.cfg.n_contexts).float().to(self.gate.weight.device)
        
        try:
            # Compute gate outputs
            logits = self.gate(c_onehot)
            
            # Apply activation function
            if self.use_softmax:
                w = F.softmax(logits / self.temperature, dim=-1)
            else:
                raw = F.relu(logits / self.temperature)
                w = raw / (raw.sum(-1, keepdim=True) + 1e-8)
            
            # 如果是序列输入，恢复为 [H, B, n_experts] 形状
            if obs.dim() > 2:
                H, B, _ = obs.shape
                w = w.view(H, B, self.n_experts)
            
            # Ensure result is on the right device
            return w.to(obs.device)
            
        except Exception as e:
            print(f"[ERROR] Exception in get_expert_weights: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return uniform weights as fallback
            if obs.dim() > 2:
                H, B, _ = obs.shape
                return torch.ones(H, B, self.n_experts, device=obs.device) / self.n_experts
            else:
                batch_size = obs.shape[0]
                return torch.ones(batch_size, self.n_experts, device=obs.device) / self.n_experts
    
    def __repr__(self):
        """自定义打印方法"""
        return (f"MooreTaskEncoder(obs_dim={self.obs_dim}, "
                f"latent_dim={self.latent_dim}, "
                f"hidden_dim={self.hidden_dim}, "
                f"n_experts={self.n_experts}, "
                f"temperature={self.temperature}, "
                f"use_softmax={self.use_softmax})") 