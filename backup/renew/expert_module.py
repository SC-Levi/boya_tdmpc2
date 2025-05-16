"""
Moore架构的核心层实现

包含混合专家架构所需的关键组件：
- 任务嵌入模块
- 专家网络结构
- 正交化处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.profiler import record_function

# Add debugging flag
DEBUG_CPU = False

# Add debug print function
def debug_print(msg, tensors=None):
    # All print statements are commented out to disable debug output
    # If tensors:
    #     shapes = {k: v.shape if hasattr(v, 'shape') else v for k, v in tensors.items()}
    #     print(f"[DEBUG] {msg} - Shapes: {shapes}")
    # else:
    #     print(f"[DEBUG] {msg}")
    pass


class InputLayer(nn.Module):
    """
    将输入复制到多个专家处理的层
    """
    def __init__(self, n_models=4):
        super().__init__()
        self.n_models = n_models

    def forward(self, x):
        # 对每个专家复制相同的输入
        # 支持多种输入维度:
        # - 1D输入: [input_dim] -> [n_models, 1, input_dim]
        # - 2D输入: [batch_size, input_dim] -> [n_models, batch_size, input_dim]
        # - 3D输入: [seq_len, batch_size, input_dim] -> [n_models, seq_len, batch_size, input_dim]
        # - 3D输入: [batch_size, seq_len, input_dim] -> [n_models, batch_size, seq_len, input_dim]
        
        # 检查输入维度并处理
        if x.dim() == 1:  # [input_dim]
            return x.unsqueeze(0).unsqueeze(0).repeat(self.n_models, 1, 1)
        elif x.dim() == 2:  # [batch_size, input_dim]
            return x.unsqueeze(0).repeat(self.n_models, 1, 1)
        elif x.dim() == 3:  # 3D输入: [seq_len/batch_size, batch_size/seq_len, input_dim]
            return x.unsqueeze(0).repeat(self.n_models, 1, 1, 1)
        elif x.dim() == 4:  # 已经是4D: [?, n_models, ?, ?]
            if x.size(1) == self.n_models:
                return x  # 已经有正确的形状，直接返回
            else:
                return x.unsqueeze(1).repeat(1, self.n_models, 1, 1)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}, shape: {x.shape}")



class ParallelLayer(nn.Module):
    """
    专家并行处理层
    """
    def __init__(self, layer_structure):
        super().__init__()
        self.layer = layer_structure

    def forward(self, x):
        # 输入: [n_models, batch_size, input_dim]
        # 对每个专家单独执行前向传播
        # 输出: [n_models, batch_size, output_dim]
        batch_size = x.shape[1]
        n_models = x.shape[0]
        
        # 重塑为单个大批量
        reshaped_x = x.reshape(n_models * batch_size, -1)
        
        # 通过网络层传递
        output = self.layer(reshaped_x)
        
        # 重塑回正确的形状
        return output.view(n_models, batch_size, -1)


class OrthogonalLayer1D(nn.Module):
    """
    使用Gram-Schmidt正交化处理专家输出
    
    该层实现了批量Gram-Schmidt正交化过程，确保各专家输出相互正交。
    正交化过程保留梯度流，允许端到端训练。
    
    Args:
        eps: 数值稳定性的小常数，避免除零错误
        normalize_input: 是否在正交化前先归一化输入
        normalize_output: 是否对正交化后的输出进行归一化
    """
    def __init__(self, eps=1e-8, normalize_input=True, normalize_output=True):
        super().__init__()
        self.eps = eps
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

    def forward(self, x):
        """
        对专家输出进行正交化
        
        Args:
            x: 专家输出张量 [n_models, batch_size, output_dim]
            
        Returns:
            正交化后的专家输出 [n_models, batch_size, output_dim]
        """
        n_models, batch_size, output_dim = x.shape
        
        # 将形状调整为 [batch_size, n_models, output_dim]
        x_permuted = x.permute(1, 0, 2).contiguous()
        
        # 初始化结果张量
        q = torch.zeros_like(x_permuted)
        
        # 批量处理
        for b in range(batch_size):
            x_batch = x_permuted[b]  # [n_models, output_dim]
            q_batch = []
            
            # 第一个向量：归一化
            v1 = x_batch[0].clone()  # Explicitly clone to avoid modifying original
            norm_v1 = torch.norm(v1, p=2)
            if self.normalize_input and norm_v1 > self.eps:
                v1 = v1 / norm_v1  # Creates new tensor
            q_batch.append(v1)
            
            # 增强的Gram-Schmidt正交化过程
            for i in range(1, n_models):
                vi = x_batch[i].clone()
                
                # 迭代两次Gram-Schmidt以提高正交性
                for _ in range(2):  # 重复正交化过程以提高精度
                    # 减去所有之前向量的投影
                    for j in range(len(q_batch)):
                        qj = q_batch[j]
                        
                        # 确保qj已归一化
                        qj_norm = torch.norm(qj)
                        if qj_norm > self.eps:
                            qj_normalized = qj / qj_norm  # Creates new tensor
                        else:
                            qj_normalized = qj
                        
                        # 计算投影系数 (vi·qj)
                        dot_product = torch.sum(vi * qj_normalized)
                        
                        # 减去投影 - FIXED: explicitly use non-in-place operation
                        vi = vi - dot_product * qj_normalized  # Creates new tensor
                
                # 最终归一化
                vi_norm = torch.norm(vi, p=2)
                if self.normalize_output and vi_norm > self.eps:
                    vi = vi / vi_norm  # Creates new tensor
                elif vi_norm <= self.eps:
                    # 如果向量接近零，生成随机正交向量
                    random_v = torch.randn_like(vi)
                    # 对之前所有向量做正交化
                    for j in range(len(q_batch)):
                        qj = q_batch[j]
                        qj_norm = torch.norm(qj)
                        if qj_norm > self.eps:
                            qj_normalized = qj / qj_norm  # Creates new tensor
                            dot_product = torch.sum(random_v * qj_normalized)
                            # FIXED: explicitly use non-in-place operation
                            random_v = random_v - dot_product * qj_normalized  # Creates new tensor
                    
                    random_norm = torch.norm(random_v)
                    if random_norm > self.eps:
                        vi = random_v / random_norm  # Creates new tensor
                    else:
                        # 极端情况：使用单位向量 - 避免原地操作
                        zero_tensor = torch.zeros_like(vi)
                        # 创建单位向量
                        unit_vector = zero_tensor.clone()
                        unit_vector_shape = list(unit_vector.shape)
                        if len(unit_vector_shape) > 0:  # 防止零维张量
                            # 如果是一维以上的张量，用索引设置第一个元素
                            mask = torch.zeros_like(unit_vector)
                            mask.view(-1)[0] = 1.0
                            unit_vector = mask  # Assign new tensor
                        vi = unit_vector
                
                q_batch.append(vi)
            
            # 将结果保存到输出张量 - use copy instead of in-place assignment
            q[b] = torch.stack(q_batch)
        
        # 转回原始形状 [n_models, batch_size, output_dim]
        return q.permute(1, 0, 2).contiguous()
        
    def extra_repr(self):
        return f'eps={self.eps}, normalize_input={self.normalize_input}, normalize_output={self.normalize_output}'


class MooreExpertsModule(nn.Module):
    """
    Moore混合专家模块，包含多个专家网络和正交化
    优化版本：使用批处理矩阵运算和专用层提高效率
    """
    def __init__(self, input_dim, hidden_dims, output_dim, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 假设 hidden_dims=[H], 这里只做两层：D→H→L
        H = hidden_dims[-1] if len(hidden_dims)>0 else output_dim
        # 第一层专家权重 bias： E × D → H
        self.expert_w1 = nn.Parameter(torch.empty(n_experts, input_dim, H))
        self.expert_b1 = nn.Parameter(torch.zeros(n_experts, H))
        # 第二层专家权重 bias： E × H → L
        self.expert_w2 = nn.Parameter(torch.empty(n_experts, H, output_dim))
        self.expert_b2 = nn.Parameter(torch.zeros(n_experts, output_dim))
        # 初始化
        nn.init.xavier_uniform_(self.expert_w1.view(n_experts, -1),
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.expert_w2.view(n_experts, -1),
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, input_dim]
        return : [E, B, output_dim]
        ——改进点——
        1. 去掉 repeat / bmm，全部用 einsum；显存占用更低、速度更快
        2. 激活直接就地 relu_，避免一次额外拷贝
        """
        E, D, H = self.expert_w1.shape          # expert_w1 : [E,D,H]
        B = x.size(0)

        # 第一层：x @ W1 + b1   得到 [B,E,H]
        h = torch.einsum('bd,edh->beh', x, self.expert_w1).add_(self.expert_b1).relu_()

        # 第二层：h @ W2 + b2    得到 [B,E,L]
        out = torch.einsum('beh,ehl->bel', h, self.expert_w2).add_(self.expert_b2)

        # 转回 [E,B,L]
        return out.permute(1, 0, 2).contiguous()

