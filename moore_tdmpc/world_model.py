import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import time
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast
from torch.profiler import record_function

from moore_tdmpc.task_encoder import MooreTaskEncoder
from moore_tdmpc.world_model_core import MoETransitionRewardModel

# Check if torch.compile is available (PyTorch 2.0+)
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')

# Utility functions
def normalize_bt(x, name):
    """
    Normalize tensor to batch-first format [B,T,...].
    
    Args:
        x: Input tensor, can be 2D [B,D] or 3D [B,T,D]
        name: Name of tensor for error messages
        
    Returns:
        Tensor in [B,T,...] format
    """
    if x.dim() == 2:
        return x.unsqueeze(1)  # [B,D] -> [B,1,D]
    if x.dim() == 3:
        return x  # Already [B,T,D]
    raise ValueError(f"{name} must be 2D or 3D, got shape {tuple(x.shape)} with {x.dim()} dimensions")

class MooreWorldModel(nn.Module):
    """
    Moore World Model: 一个完全自包含的多任务世界模型，实现混合专家方法进行多任务强化学习
    
    使用混合专家方法实现多任务强化学习的世界模型，包括：
    1. 任务编码器：生成任务特定的潜在表示
    2. 混合专家动力学模型：预测下一个状态
    3. 混合专家奖励模型：预测奖励
    4. 策略先验：生成动作
    5. Q网络集成：评估状态动作值
    
    每个组件都使用InputLayer、ParallelLayer和OrthogonalLayer1D实现专家输入复制、并行处理和正交化。
    """
    
    def __init__(self, cfg):
        """
        初始化Moore世界模型
        
        Args:
            cfg: 包含模型参数的配置对象
        """
        super().__init__()
        
        # 验证Moore特定参数
        assert hasattr(cfg, 'n_experts'), "配置必须指定n_experts参数"
        assert hasattr(cfg, 'temperature'), "配置必须指定temperature参数"
        assert hasattr(cfg, 'obs_dim'), "配置必须指定obs_dim参数"
        assert hasattr(cfg, 'action_dim'), "配置必须指定action_dim参数"
        assert hasattr(cfg, 'latent_dim'), "配置必须指定latent_dim参数"
        
        # 存储Moore特定参数
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.temperature = cfg.temperature
        self.moore_temperature = getattr(cfg, 'moore_temperature', self.temperature)
        self.use_softmax = getattr(cfg, 'use_softmax', True)
        
        # 存储维度
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.latent_dim = cfg.latent_dim
        self.hidden_dim = getattr(cfg, 'hidden_dim', 256)
        self.device = getattr(cfg, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 注册标准差限制参数
        self.log_std_min = getattr(cfg, 'log_std_min', -5)
        self.log_std_max = getattr(cfg, 'log_std_max', 2)
        self.log_std_dif = self.log_std_max - self.log_std_min


        
        # 初始化嵌入缓存
        self._cached_features = None
        
        # 启用梯度检查点
        self.use_checkpoint = getattr(cfg, 'use_checkpoint', False)
        
        # 初始化所有组件
        self._initialize_components()
        
        # ────────── ✅ 仅此一次 torch.compile ──────────
        if TORCH_COMPILE_AVAILABLE and getattr(cfg, "compile_dynamics", True):
            # 动态维度只有 batch_size，可固定 horizon=cfg.horizon(=3)
            self._dyn_vec = torch.compile(
                self.dynamics_vectorized,           # 要编译的函数
                dynamic=False,                      # 关闭动态形状 → 减少 Overhead
                mode="reduce-overhead"              # Inductor 自带 profile-guided 优化
            )
            # 用编译后的版本替换原方法
            self.dynamics_vectorized = self._dyn_vec
            print("[MooreWorldModel] dynamics_vectorized compiled once (Inductor-cached)")

        # ========= Action-mask for variable-length action spaces (offline mt30 / mt80) =========
        if getattr(self.cfg, "multitask", False) and hasattr(self.cfg, "action_dims"):
            n_tasks     = len(self.cfg.tasks)          # e.g. 30 or 80
            action_dim  = self.cfg.action_dim          # 全局最大动作维
            mask        = torch.zeros(n_tasks, action_dim)

            # cfg.action_dims 必须与 cfg.tasks 顺序一一对应
            for i, real_dim in enumerate(self.cfg.action_dims):
                mask[i, :real_dim] = 1.

            # 注册为 buffer，随模型走 device
            self.register_buffer("_action_masks", mask)
        else:
            # 单任务或未指定 action_dims → 全 1 mask
            self.register_buffer("_action_masks", torch.ones(1, self.cfg.action_dim))

        # 将模型移至指定设备
        self.to(self.device)
        
        # 计算并存储总参数
        self._total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def total_params(self):
        return self._total_params
    
    def _initialize_components(self):
        """初始化模型所有组件"""
        # 获取混合精度配置
        use_mixed_precision = getattr(self.cfg, 'use_mixed_precision', False)
        if use_mixed_precision:
            print("启用混合精度训练 (AMP) 在 MooreWorldModel 中")
            
        # 任务编码器：生成特定于上下文的潜在表示
        self.task_encoder = MooreTaskEncoder(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_experts=self.n_experts,
            temperature=self.moore_temperature,
            use_softmax=self.use_softmax,
            use_checkpoint=self.use_checkpoint
        )
        
        # 获取 top_k 参数，默认为 1
        reward_top_k = getattr(self.cfg, 'reward_top_k', 1)
        
        # 使用共享专家的组合动力学+奖励模型
        self.core = MoETransitionRewardModel(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            n_experts=self.n_experts,
            temperature=self.temperature,
            use_softmax=self.use_softmax,
            reward_dim=101,  # TD-MPC2默认使用101个奖励箱
            use_mixed_precision=use_mixed_precision
        )
        
        # 策略先验：使用简单的MLP网络
        self._pi = self._create_mlp(
            input_dim=self.latent_dim,
            hidden_dims=[self.hidden_dim, self.hidden_dim],
            output_dim=2*self.action_dim  # 输出均值和对数标准差
        )
        
        # 创建Q网络集成
        self._initialize_q_networks()
    
    def _create_mlp(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        """创建简单的MLP网络"""
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 不在最后一层添加激活函数
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        return nn.Sequential(*layers)
    
    def _initialize_q_networks(self):
        """初始化Q网络集成"""
        # 从配置获取参数
        mlp_dim = getattr(self.cfg, 'mlp_dim', self.hidden_dim)
        num_bins = getattr(self.cfg, 'num_bins', 1)
        dropout = getattr(self.cfg, 'dropout', 0.0)
        num_q = getattr(self.cfg, 'num_q', 2)
        
        # 创建多个Q网络
        self._q_networks = nn.ModuleList([
            self._create_mlp(
                input_dim=self.latent_dim + self.action_dim,  # 只接受潜在表示和动作，不包含任务维度
                hidden_dims=[mlp_dim, mlp_dim],
                output_dim=max(num_bins, 1),
                dropout=dropout
            ) for _ in range(num_q)
        ])
        
        # 创建目标Q网络
        self._target_q_networks = nn.ModuleList([
            copy.deepcopy(q_net) for q_net in self._q_networks
        ])
        
        # 设置目标网络为评估模式并禁用梯度
        for q_net in self._target_q_networks:
            q_net.eval()
            for param in q_net.parameters():
                param.requires_grad = False
    
    def encode(self, obs, task=None):
        # 1) 不是“真正的”批量轨迹，就当作单帧
        #    也包括所有高维度的图像输入
        if obs.ndim != 3 or obs.shape[1] == 1:
            # squeeze 掉长度为1的 time 维，或者直接当 [B, ...] 处理
            x = obs.squeeze(1) if obs.ndim == 3 else obs
            # flatten 后面所有维度到 feature 维
            B = x.size(0)
            f = int(torch.tensor(x.shape[1:]).prod().item())
            x = x.reshape(B, f)
            # 走原始 encoder
            out = self.task_encoder(x, task) if task is not None else self.task_encoder(x)
            v = out[0] if isinstance(out, tuple) else out  # [B, D_latent]
            return v

        # 2) 真正的批量轨迹：[B, T+1, D_obs]
        #    normalize_bt 会自动把 time-first → batch-first
        obs_bt = normalize_bt(obs, 'obs')   # [B, Tp1, D_obs]
        B, Tp1, D_obs = obs_bt.shape
        flat = obs_bt.reshape(B * Tp1, D_obs)

        if task is not None:
            task_flat = task.view(B,1).expand(B, Tp1).reshape(-1)
            out = self.task_encoder(flat, task_flat)
        else:
            out = self.task_encoder(flat)

        v_flat = out[0] if isinstance(out, tuple) else out    # [B*Tp1, D_latent]
        D_latent = v_flat.shape[-1]
        return v_flat.view(B, Tp1, D_latent)


    
    def dynamics_step(self, z, a, task=None):
        """
        使用混合专家模型预测下一个潜在状态和专家权重
        
        Args:
            z: 当前潜在状态 [batch_size, latent_dim]
            a: 动作 [batch_size, action_dim]
            task: 任务标识符（可选）
            
        Returns:
            next_z: 下一个潜在状态 [batch_size, latent_dim]
            expert_weights: 专家权重 [batch_size, n_experts]
        """
        with record_function("WorldModel_dynamics_step"):
            # 使用梯度检查点包装动力学模型前向传播
            if self.use_checkpoint and self.training:
                # 使用checkpoint时，只获取next_z，丢弃reward和权重信息
                next_z, _, (expert_weights, _) = checkpoint(self.core, z, a, task)
            else:
                # 获取next_z和权重，但丢弃reward
                next_z, _, (expert_weights, _) = self.core(z, a, task)
            
            return next_z, expert_weights
    
    def reward(self, z, a, z_next=None, task=None):
        """
        预测给定潜在状态和动作的奖励
        
        Args:
            z: 潜在状态 [batch_size, latent_dim] 或 [batch_size, horizon, latent_dim]
            a: 动作 [batch_size, action_dim] 或 [batch_size, horizon, action_dim]
            z_next: 下一个潜在状态（未使用）
            task: 任务标识符（可选）
            
        Returns:
            reward_dist: 奖励分布 [batch_size, reward_dim] 或 [batch_size, horizon, reward_dim]
            expert_weights: 专家权重 [batch_size, n_experts] 或 [batch_size, horizon, n_experts]
        """
        with record_function("WorldModel_reward"):
            # 判断是否在训练模式下使用梯度检查点
            if self.use_checkpoint and self.training:
                # 使用梯度检查点包装奖励模型前向传播
                def _reward_forward(z_input, a_input, task_input):
                    # 使用shared core，只需要reward和其权重
                    _, reward, (_, expert_weights) = self.core(z_input, a_input, task_input)
                    return reward, expert_weights
                
                # 根据输入维度进行不同处理
                if z.dim() == 3 and a.dim() == 3:
                    # 处理序列输入 [B,H,D] 和 [B,H,A]
                    B, H, _ = z.shape
                    
                    # 需要处理序列输入的情况
                    # 如果序列比较长，可能需要进一步优化以减少内存消耗
                    rewards = []
                    expert_weights_list = []
                    
                    for t in range(H):
                        z_t = z[:, t]  # [B, D]
                        a_t = a[:, t]  # [B, A]
                        reward_t, weights_t = checkpoint(_reward_forward, z_t, a_t, task)
                        rewards.append(reward_t)
                        expert_weights_list.append(weights_t)
                    
                    # 堆叠沿时间维度的输出
                    reward_dist = torch.stack(rewards, dim=1)  # [B, H, reward_dim]
                    expert_weights = torch.stack(expert_weights_list, dim=1) if expert_weights_list[0] is not None else None
                else:
                    # 标准2D输入
                    reward_dist, expert_weights = checkpoint(_reward_forward, z, a, task)
            else:
                # 非checkpoint情况，直接调用
                if z.dim() == 3 and a.dim() == 3:
                    # 处理序列输入 [B,H,D] 和 [B,H,A]
                    B, H, _ = z.shape
                    
                    # 需要处理序列输入的情况
                    rewards = []
                    expert_weights_list = []
                    
                    for t in range(H):
                        z_t = z[:, t]  # [B, D]
                        a_t = a[:, t]  # [B, A]
                        _, reward_t, (_, weights_t) = self.core(z_t, a_t, task)
                        rewards.append(reward_t)
                        expert_weights_list.append(weights_t)
                    
                    # 堆叠沿时间维度的输出
                    reward_dist = torch.stack(rewards, dim=1)  # [B, H, reward_dim]
                    expert_weights = torch.stack(expert_weights_list, dim=1)
                else:
                    # 标准2D输入
                    _, reward_dist, (_, expert_weights) = self.core(z, a, task)
            
            return reward_dist, expert_weights
    
    def policy(self, z, task=None):
        """
        预测给定潜在状态的策略（动作分布参数）
        
        Args:
            z: 潜在状态 [batch_size, latent_dim]
            task: 已不再使用，保留参数仅为兼容性
            
        Returns:
            动作分布的(均值, 标准差)元组
        """
        # 检查输入是否已经是潜在表示维度
        assert z.shape[-1] == self.latent_dim, f"Expected latent dimension {self.latent_dim}, got {z.shape[-1]}"

        # 输入形状调试打印
        if getattr(self.cfg, 'debug', False):
            print(f"Policy input shape - z: {z.shape}")
        
        # 获取均值和对数标准差
        pi_output = self._pi(z)
        mean, log_std = pi_output.chunk(2, dim=-1)
        
        # 应用对数标准差的限制
        log_std_min = getattr(self.cfg, 'log_std_min', -5)
        log_std_max = getattr(self.cfg, 'log_std_max', 2)
        log_std_dif = log_std_max - log_std_min
        
        # 限制对数标准差在[log_std_min, log_std_max]范围内
        log_std = log_std_min + 0.5 * (1 + torch.tanh(log_std)) * log_std_dif
        std = torch.exp(log_std)
        
        # 输出形状调试打印
        if getattr(self.cfg, 'debug', False):
            print(f"Policy output shapes - mean: {mean.shape}, std: {std.shape}")
        
        return mean, std
    
    def pi(self, z, task=None):
        """
        从策略先验中采样动作
        
        Args:
            z: 潜在状态 [batch_size, latent_dim]
            task: 已不再使用，保留参数仅为兼容性
            
        Returns:
            采样动作 [batch_size, action_dim] 和信息字典的元组
        """
        # 如果 z 是 tuple，取第一个元素
        if isinstance(z, tuple):
            z = z[0]
        
        # 检查输入是否已经是潜在表示维度
        assert z.shape[-1] == self.latent_dim, f"Expected latent dimension {self.latent_dim}, got {z.shape[-1]}"
        
        # 获取策略参数
        mean, std = self.policy(z)
        
        # 创建正态分布
        dist = torch.distributions.Normal(mean, std)
        
        # 采样
        eps = torch.randn_like(mean)
        action = mean + std * eps
        
        # 计算动作熵（包含梯度）
        entropy = dist.entropy()
        
        # 裁剪到有效范围
        action = action.clamp(-1, 1)
        
        # 构造信息字典
        info = {
            "mean": mean,
            "std": std,
            "log_std": torch.log(std),  # 添加log_std用于兼容性
            "action_prob": 1.0,
            "entropy": entropy  # 添加熵
        }
        
        return action, info
    
    def sample_action(self, z, temp=1.0, noise=None, task=None):
        """
        从给定潜在状态的策略中采样动作
        
        Args:
            z: 潜在状态 [batch_size, latent_dim]
            temp: 采样的温度参数
            noise: 可选的添加到分布的噪声
            task: 已不再使用，保留参数仅为兼容性
            
        Returns:
            采样动作 [batch_size, action_dim]
        """
        # 检查输入是否已经是潜在表示维度
        assert z.shape[-1] == self.latent_dim, f"Expected latent dimension {self.latent_dim}, got {z.shape[-1]}"
        
        # 获取策略参数
        mean, std = self.policy(z)
        std = std * temp
        
        # 采样动作
        if noise is None:
            noise = torch.randn_like(mean)
        action = mean + std * noise
        
        # 将动作裁剪到有效范围
        return action.clamp(-1, 1)
    
    def dynamics(self, z, actions, task=None):
        """
        给定初始状态和动作序列，预测潜在状态序列
        
        Args:
            z: 初始潜在状态 [batch_size, latent_dim]
            actions: 动作序列 [batch_size, horizon, action_dim]
            task: 已不再使用，保留参数仅为兼容性
            
        Returns:
            预测的潜在状态序列 [batch_size, horizon, latent_dim]
            专家权重 [batch_size, horizon, n_experts]
        """
        # 检查输入是否已经是潜在表示维度
        assert z.shape[-1] == self.latent_dim, f"Expected latent dimension {self.latent_dim}, got {z.shape[-1]}"
        
        horizon = actions.shape[1]
        batch_size = z.shape[0]
        
        # 初始化潜在状态序列
        z_seq = torch.zeros(batch_size, horizon, self.latent_dim, device=z.device)
        expert_weights_list = []
        
        # 逐步预测潜在状态
        z_current = z
        for t in range(horizon):
            a_t = actions[:, t]
            z_next, expert_weights = self.dynamics_step(z_current, a_t)
            z_seq[:, t] = z_next
            expert_weights_list.append(expert_weights)
            z_current = z_next
        
        # 堆叠专家权重
        if expert_weights_list:
            expert_weights = torch.stack(expert_weights_list, dim=1)  # [batch_size, horizon, n_experts]
        else:
            # 如果没有权重，创建一个与z_seq形状一致的零张量
            expert_weights = torch.zeros(batch_size, horizon, self.n_experts, device=z.device)
        
        return z_seq, expert_weights
    
    def next(self, z, a, task=None):
        """
        预测下一个潜在状态
        
        Args:
            z: 当前潜在状态 [batch_size, latent_dim]
            a: 动作 [batch_size, action_dim]
            task: 已不再使用，保留参数仅为兼容性
            
        Returns:
            下一个潜在状态 [batch_size, latent_dim]
        """
        # 检查输入是否已经是潜在表示维度
        assert z.shape[-1] == self.latent_dim, f"Expected latent dimension {self.latent_dim}, got {z.shape[-1]}"
        
        # dynamics_step 返回 (next_z, expert_weights)
        next_z, _ = self.dynamics_step(z, a, task)
        return next_z
    
    def Q(self, z, a, task=None, return_type='min', target=False, detach=False):
        """
        计算Q值 - 给定状态动作对的预期回报
        
        Args:
            z: 潜在状态 [batch_size, latent_dim] 或 [batch_size, time, latent_dim]
            a: 动作 [batch_size, action_dim] 或 [batch_size, time, action_dim]
            task: 已不再使用，保留参数仅为兼容性
            return_type: 返回类型：'min'（最小值），'mean'（平均值），'all'（所有值）
            target: 是否使用目标网络
            detach: 是否分离梯度
            
        Returns:
            q_value: Q值 [batch_size, 1]、[batch_size, time, 1] 或相应的所有Q值
        """
        with record_function("WorldModel_Q"):
            # 支持 [B,T,D] + [B,T,A] 的序列输入：flatten → 调用 → reshape
            if z.dim() == 3 and a.dim() == 3:
                B, T, D = z.shape
                _, _, A = a.shape
                # flatten time
                z_flat = z.reshape(B*T, D)
                a_flat = a.reshape(B*T, A)
                # 递归调用一维情形
                q_flat = self.Q(z_flat, a_flat, task=task,
                               return_type=return_type,
                               target=target, detach=detach)
                # q_flat: [B*T, 1] 或 [B*T, num_q]
                # reshape 回 [B, T, ...]
                if return_type in ('min','mean'):
                    return q_flat.reshape(B, T, 1)
                elif return_type == 'all':
                    # 返回 list of [B*T, ?] → list of [B,T,?]
                    return [q_.reshape(B, T, -1) for q_ in q_flat]
                else:
                    raise ValueError(f"无效的 return_type {return_type}")

            # 原来的 2D 分支
            assert z.dim() == 2 and a.dim() == 2, \
                f"Q网络期望2D输入，收到：z{z.shape}, a{a.shape}"
            
            # 检查批次大小匹配
            assert z.shape[0] == a.shape[0], \
                f"批次大小不匹配：z{z.shape[0]}, a{a.shape[0]}"
                
            # 连接潜在表示和动作
            za = torch.cat([z, a], dim=-1)
            
            # 选择Q网络：原始或目标网络
            q_networks = self._target_q_networks if target else self._q_networks
            
            # 获取所有Q网络的输出
            q_values = []
            for q_net in q_networks:
                q = q_net(za)
                
                if detach:
                    q = q.detach()
                    
                q_values.append(q)
                
            # 根据返回类型处理
            if return_type == 'min':
                # 返回所有Q值的最小值
                q_cat = torch.cat(q_values, dim=1)
                q_min, _ = torch.min(q_cat, dim=1, keepdim=True)
                return q_min
            elif return_type in ('mean', 'avg'):
                # 返回所有Q值的平均值
                q_cat = torch.cat(q_values, dim=1)
                q_mean = torch.mean(q_cat, dim=1, keepdim=True)
                return q_mean
            elif return_type == 'all':
                # 返回所有Q值
                return q_values
            else:
                raise ValueError(f"无效的返回类型: {return_type}")
    
    def soft_update_target_Q(self, tau=0.005):
        """
        使用Polyak平均软更新目标Q网络
        
        Args:
            tau: 软更新参数，决定有多少比例的当前网络权重被混合进目标网络
        """
        for target_net, current_net in zip(self._target_q_networks, self._q_networks):
            for target_param, current_param in zip(target_net.parameters(), current_net.parameters()):
                target_param.data.copy_(
                    (1 - tau) * target_param.data + tau * current_param.data
                )
    
    def train(self, mode=True):
        """
        重写train方法以保持目标Q网络处于评估模式
        """
        super().train(mode)
        if mode:
            # 确保目标Q网络始终处于评估模式
            for q_net in self._target_q_networks:
                q_net.eval()
        return self
    
    def __repr__(self):
        """自定义打印模型信息"""
        repr_str = f"MooreWorldModel(total_params={self._total_params}, "
        repr_str += f"latent_dim={self.latent_dim}, "
        repr_str += f"action_dim={self.action_dim}, "
        repr_str += f"n_experts={self.n_experts}, "
        repr_str += f"temperature={self.temperature})\n"
        
        repr_str += f"Task Encoder: {self.task_encoder}\n"
        repr_str += f"Transition-Reward Core: {self.core}\n"
        repr_str += f"Policy prior: {self._pi}\n"
        repr_str += f"Q-functions: Ensemble(n_models={len(self._q_networks)})\n"
        
        return repr_str
    def step(self, z, a, task=None):
        """
        兼容TD-MPC2原有的step接口：
         - 先用dynamics_step预测下一个潜在状态
         - 再用policy生成动作分布的参数
        Args:
            z: [batch, latent_dim] 当前潜在状态
            a: [batch, action_dim]  上一步动作（这里并不用于 sampling，仅为兼容原接口）
            task: （兼容）任务ID
        Returns:
            next_z: [batch, latent_dim]
            mu:     [batch, action_dim]
            log_std:[batch, action_dim]
            log_pi: [batch] 动作的对数概率
        """
        # dynamics
        next_z, _ = self.dynamics_step(z, a, task)

        # policy on next state
        mu, std = self.policy(next_z, task)
        log_std = torch.log(std)

        # 计算动作的对数概率
        dist = torch.distributions.Normal(mu, std)
        # 对所有action维度的log_prob求和
        log_pi = dist.log_prob(mu).sum(-1)

        return next_z, mu, log_std, log_pi
        
    def dynamics_vectorized(self, z_seq, a_seq):
        """Predicts a sequence of latent states given an initial latent state and action sequence.
        
        Automatically handles different input formats:
        1. z_seq shape [B,D] and a_seq shape [B,T,A] -> return [B,T,D]
        2. z_seq shape [B,D] and a_seq shape [T,B,A] -> return [B,T,D]
        3. z_seq shape [B,D] and a_seq shape [B,A] -> return [B,D]
        
        Args:
            z_seq: Initial latent state tensor [B,D]
            a_seq: Action sequence tensor [B,T,A], [T,B,A], or [B,A]
            
        Returns:
            next_z: Sequence of predicted latent states [B,T,D]
            expert_weights: Expert weights for each prediction [B,T,n_experts]
        """
        # Special case for single-step prediction (2D action)
        if z_seq.dim() == 2 and a_seq.dim() == 2:
            # Check batch dimension matches
            assert z_seq.shape[0] == a_seq.shape[0], f"Batch dimension mismatch: z_seq={z_seq.shape}, a_seq={a_seq.shape}"
            # Call single-step dynamics
            return self.dynamics_step(z_seq, a_seq)
        
        # Ensure z_seq is [B,D]
        assert z_seq.dim() == 2, f"z_seq must be 2D [B,D], got {z_seq.shape}"
        
        # Detect action sequence format and standardize to batch-first [B,T,A]
        if a_seq.dim() == 3:
            # Determine if input is [B,T,A] or [T,B,A]
            if a_seq.shape[0] > a_seq.shape[1] or a_seq.shape[0] == z_seq.shape[0]:
                # Already [B,T,A]
                a_bt = a_seq
            else:
                # Format is [T,B,A], convert to [B,T,A]
                a_bt = a_seq.permute(1, 0, 2).contiguous()
                
            if self.cfg.debug_mode:
                print(f"[DEBUG] dynamics_vectorized: input z_seq={z_seq.shape}, a_seq={a_seq.shape}, normalized a_bt={a_bt.shape}")
            
            # Verify batch dimension matches
            assert z_seq.shape[0] == a_bt.shape[0], (
                f"Batch dimension mismatch after normalization: "
                f"z_seq={z_seq.shape[0]}, a_bt={a_bt.shape[0]}"
            )
        else:
            # Handle 2D action by adding time dimension: [B,A] -> [B,1,A]
            a_bt = a_seq.unsqueeze(1)
            if self.cfg.debug_mode:
                print(f"[DEBUG] dynamics_vectorized: expanded 2D action {a_seq.shape} to {a_bt.shape}")
        
        # Get dimensions
        batch_size, seq_len, action_dim = a_bt.shape
        
        # Performance optimization: vectorized processing for MoE models
        if hasattr(self, 'n_experts'):
            # Optimization for longer sequences
            z_next_list = []
            weights_list = []
            
            # Process the sequence step by step
            z_current = z_seq
            for t in range(seq_len):
                a_t = a_bt[:, t]  # [B,A]
                z_next, weights = self.dynamics_step(z_current, a_t)
                z_next_list.append(z_next)
                weights_list.append(weights)
                z_current = z_next  # Update for next iteration
            
            # Stack results along time dimension
            z_next = torch.stack(z_next_list, dim=1)  # [B,T,D]
            expert_weights = torch.stack(weights_list, dim=1)  # [B,T,n_experts]
            
            return z_next, expert_weights
        else:
            # Fallback for older implementations without n_experts
            z_next_list = []
            
            # Process the sequence
            z_current = z_seq
            for t in range(seq_len):
                a_t = a_bt[:, t]
                z_next, _ = self.dynamics_step(z_current, a_t)
                z_next_list.append(z_next)
                z_current = z_next
                
            # Stack results
            z_next = torch.stack(z_next_list, dim=1)  # [B,T,D]
            # Create dummy expert weights
            dummy_weights = torch.ones(batch_size, seq_len, 1, device=z_next.device)
            
            return z_next, dummy_weights
            
    def core_vectorized(self, z_seq, a_seq, task=None):
        """
        预测从初始潜在状态和动作序列生成的潜在状态和奖励序列
        
        自动处理不同的输入格式:
        1. z_seq shape [B,D] and a_seq shape [B,T,A] -> return [B,T,D], [B,T,R], [B,T,E], [B,T,E]
        2. z_seq shape [B,D] and a_seq shape [T,B,A] -> return [B,T,D], [B,T,R], [B,T,E], [B,T,E]
        3. z_seq shape [B,D] and a_seq shape [B,A] -> return [B,D], [B,R], [B,E], [B,E]
        
        Args:
            z_seq: 初始潜在状态张量 [B,D]
            a_seq: 动作序列张量 [B,T,A], [T,B,A], 或 [B,A]
            task: 可选的任务ID
            
        Returns:
            next_z_seq: 预测的潜在状态序列 [B,T,D]
            reward_seq: 预测的奖励序列 [B,T,reward_dim]
            (w_dyn_seq, w_rew_seq): 专家权重序列 ([B,T,n_experts], [B,T,n_experts])
        """
        # 特殊情况：单步预测 (2D action)
        if z_seq.dim() == 2 and a_seq.dim() == 2:
            # 检查批次维度是否匹配
            assert z_seq.shape[0] == a_seq.shape[0], f"批次维度不匹配: z_seq={z_seq.shape}, a_seq={a_seq.shape}"
            # 调用单步输出
            next_z, reward, (w_dyn, w_rew) = self.core(z_seq, a_seq, task)
            return next_z, reward, (w_dyn, w_rew)
        
        # 确保z_seq是 [B,D]
        assert z_seq.dim() == 2, f"z_seq必须是2D [B,D]，当前为{z_seq.shape}"
        
        with record_function("WorldModel_core_vectorized"):
            # 使用梯度检查点包装core_vectorized调用
            if self.use_checkpoint and self.training:
                return checkpoint(self.core.core_vectorized, z_seq, a_seq, task)
            else:
                return self.core.core_vectorized(z_seq, a_seq, task)
