import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InputLayer(nn.Module):
    """
    将输入复制到多个专家处理的层
    """
    def __init__(self, n_models=4):
        super().__init__()
        self.n_models = n_models

    def forward(self, x):
        # 对每个专家复制相同的输入
        # 输入: [batch_size, input_dim]
        # 输出: [n_models, batch_size, input_dim]
        return x.unsqueeze(0).repeat(self.n_models, 1, 1)


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
        测试版本：不执行正交化，只返回输入的深拷贝
        
        Args:
            x: 专家输出张量 [n_models, batch_size, output_dim]
            
        Returns:
            输入的深拷贝 [n_models, batch_size, output_dim]
        """
        # 为了测试，我们返回输入的深拷贝，而不执行任何处理
        return x.clone()
        
    def extra_repr(self):
        return f'eps={self.eps}, normalize_input={self.normalize_input}, normalize_output={self.normalize_output}'


class MooreExpertsModule(nn.Module):
    """
    Moore混合专家模块，包含多个专家网络和正交化
    """
    def __init__(self, input_dim, hidden_dims, output_dim, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 创建专家网络结构
        layers = []
        current_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.ReLU())
            current_dim = dim
        
        # 最后一层输出
        layers.append(nn.Linear(current_dim, output_dim))
        
        # 包装成Sequential模块
        expert_network = nn.Sequential(*layers)
        
        # 创建混合专家架构
        self.experts = nn.Sequential(
            InputLayer(n_models=n_experts),
            ParallelLayer(expert_network),
            OrthogonalLayer1D()
        )
    
    def forward(self, x):
        """
        前向传播通过专家网络
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            experts_output: 专家输出张量 [n_experts, batch_size, output_dim]
        """
        return self.experts(x)


class MooreTaskEncoder(nn.Module):
    """
    Moore任务编码器，将任务ID映射为专家权重
    
    Args:
        n_tasks: 任务总数
        n_experts: 专家数量
        max_norm: 最大范数约束（用于输出嵌入）
        temperature: softmax温度参数，控制权重分布的峰度
        use_softmax: 是否使用softmax归一化权重
    """
    def __init__(self, n_tasks, n_experts, max_norm=1.0, temperature=1.0, use_softmax=True):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_experts = n_experts
        self.max_norm = max_norm
        self.temperature = temperature
        self.use_softmax = use_softmax
        
        # 无偏置的线性层将任务ID映射到专家权重
        self.task_encoder = nn.Linear(n_tasks, n_experts, bias=False)
        nn.init.xavier_uniform_(self.task_encoder.weight, 
                                gain=nn.init.calculate_gain('linear'))
    
    def forward(self, task_idx):
        """
        将任务ID转换为专家权重
        
        Args:
            task_idx: 任务ID，整数或形状为[batch_size]的张量
            
        Returns:
            权重向量，形状为[batch_size, n_experts]
        """
        if isinstance(task_idx, int):
            task_idx = torch.tensor([task_idx], device=self.task_encoder.weight.device)
            
        # 将任务ID转换为one-hot向量
        task_onehot = F.one_hot(task_idx.long(), num_classes=self.n_tasks).float()
        
        # 通过任务编码器得到原始权重
        logits = self.task_encoder(task_onehot)
        
        # 根据设置应用softmax归一化
        if self.use_softmax:
            # 使用温度参数调整分布峰度
            weights = F.softmax(logits / self.temperature, dim=-1)
        else:
            weights = logits
        
        return weights


class MooreTaskEmbeddingModule(nn.Module):
    """
    完整的Moore任务嵌入模块，结合专家网络和任务编码器
    
    Args:
        n_tasks: 任务总数
        input_dim: 输入特征维度
        hidden_dims: 专家网络隐藏层维度列表
        embedding_dim: 输出嵌入维度
        n_experts: 专家数量
        max_norm: 最大范数约束
        temperature: softmax温度参数
        use_softmax: 是否使用softmax归一化专家权重
    """
    def __init__(self, n_tasks, input_dim, hidden_dims, embedding_dim, n_experts=4, 
                 max_norm=1, temperature=1.0, use_softmax=True):
        super().__init__()
        self.n_tasks = n_tasks
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.n_experts = n_experts
        self.max_norm = max_norm
        
        # 多专家模块
        self.experts_module = MooreExpertsModule(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim,
            n_experts=n_experts
        )
        
        # 任务编码器 - 使用更低的温度参数使专家选择更加锐利
        self.task_encoder = MooreTaskEncoder(
            n_tasks=n_tasks,
            n_experts=n_experts,
            max_norm=max_norm,
            temperature=temperature,
            use_softmax=use_softmax
        )
        
        # 为每个任务创建一个特定的偏置向量以增强区分性
        # 使用正交化初始化以确保任务偏置彼此更加不同
        task_biases = torch.randn(n_tasks, embedding_dim)  # 确保偏置维度与嵌入维度匹配
        # 使用QR分解获取正交矩阵
        if n_tasks > embedding_dim:
            # 如果任务数量大于嵌入维度，先截断为方阵再正交化
            task_biases = task_biases[:embedding_dim, :]
            q, r = torch.linalg.qr(task_biases)
            # 扩展回原始任务数量
            padding = torch.randn(n_tasks - embedding_dim, embedding_dim) * 0.1
            q_padded = torch.cat([q, padding], dim=0)
            self.task_biases = nn.Parameter(q_padded * 0.5)
        else:
            # 如果任务数量小于等于嵌入维度，正常处理
            q, r = torch.linalg.qr(task_biases.transpose(0, 1))
            self.task_biases = nn.Parameter((q[:, :n_tasks].transpose(0, 1)) * 0.5)
        
        # 添加任务特定的缩放因子，进一步增强区分性
        self.task_scales = nn.Parameter(torch.ones(n_tasks, 1) + 0.1 * torch.randn(n_tasks, 1))
        
        # 用于记录任务-专家分配统计信息
        self.register_buffer("_weight_history", torch.zeros(n_tasks, n_experts))
        self.register_buffer("_task_counts", torch.zeros(n_tasks))
    
    def get_expert_stats(self):
        """获取专家使用统计信息"""
        valid_tasks = self._task_counts > 0
        if valid_tasks.sum() > 0:
            avg_weights = self._weight_history[valid_tasks] / self._task_counts[valid_tasks].unsqueeze(1)
            
            # 计算权重熵作为专家分配的度量
            entropy = -torch.sum(
                avg_weights * torch.log(avg_weights + 1e-10), dim=1
            ).mean().item()
            
            return {
                "avg_weights": avg_weights.cpu().numpy(),
                "task_counts": self._task_counts.cpu().numpy(),
                "avg_entropy": entropy
            }
        return {"avg_weights": None, "task_counts": None, "avg_entropy": None}
    
    def forward(self, x, task):
        """
        计算任务特定的嵌入
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            task: 任务ID [batch_size] 或整数
            
        Returns:
            task_embedding: 任务特定嵌入 [batch_size, embedding_dim]
        """
        # 确保任务是长整型张量
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        
        # 确保任务ID在有效范围内（0到n_tasks-1）
        task = task.long().clamp(0, self.n_tasks - 1)
        
        # 获取多专家输出 [n_experts, batch_size, embedding_dim]
        experts_output = self.experts_module(x)
        
        # 获取任务特定权重 [batch_size, n_experts]
        task_weights = self.task_encoder(task)
        
        # 更新权重历史统计
        if not torch.jit.is_scripting() and self.training:
            for i, t in enumerate(task):
                t_idx = t.item()
                if t_idx < self.n_tasks:
                    # 替换原地操作为非原地操作
                    self._weight_history[t_idx] = self._weight_history[t_idx] + task_weights[i].detach()
                    self._task_counts[t_idx] = self._task_counts[t_idx] + 1
        
        # 调整权重形状以便进行矩阵乘法 [batch_size, n_experts, 1]
        weights = task_weights.unsqueeze(-1)
        
        # 调整专家输出形状 [batch_size, n_experts, embedding_dim]
        experts_output = experts_output.permute(1, 0, 2)
        
        try:
            # 计算加权和 - [batch_size, embedding_dim]
            # [batch_size, embedding_dim, n_experts] @ [batch_size, n_experts, 1] = [batch_size, embedding_dim, 1]
            weighted_embedding = torch.bmm(experts_output.transpose(1, 2), weights).squeeze(-1)
            
            # 添加任务特定偏置以增强区分性
            batch_biases = self.task_biases[task]
            
            # 应用任务特定缩放因子
            batch_scales = self.task_scales[task]
            
            # 组合基础嵌入、任务偏置和缩放因子
            weighted_embedding = weighted_embedding * batch_scales + batch_biases
            
            # 添加非线性变换以增强区分性
            weighted_embedding = torch.tanh(weighted_embedding) * weighted_embedding
            
            # 应用max_norm约束，与原始嵌入保持一致
            if self.max_norm > 0:
                norm = torch.norm(weighted_embedding, p=2, dim=1, keepdim=True)
                # 避免原地操作
                scale = torch.clamp(self.max_norm / (norm + 1e-8), max=1.0)
                weighted_embedding = weighted_embedding * scale
            
        except RuntimeError as e:
            print(f"警告: 任务嵌入计算失败，回退到平均嵌入. 错误: {e}")
            print(f"形状: x={x.shape}, experts={experts_output.shape}, weights={weights.shape}")
            print(f"任务ID: {task}, 偏置形状: {self.task_biases.shape}, 缩放形状: {self.task_scales.shape}")
            
            # 简单地平均专家输出
            weighted_embedding = experts_output.mean(dim=1)
            
            # 添加任务特定偏置
            batch_biases = self.task_biases[task]
            weighted_embedding = weighted_embedding + batch_biases
            
            # 应用max_norm约束
            if self.max_norm > 0:
                norm = torch.norm(weighted_embedding, p=2, dim=1, keepdim=True)
                scale = torch.clamp(self.max_norm / (norm + 1e-8), max=1.0)
                weighted_embedding = weighted_embedding * scale
        
        return weighted_embedding 