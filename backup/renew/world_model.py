import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast
from torch.profiler import record_function
from torch import Tensor
from task_encoder import MooreTaskEncoder
from world_model_core import MoETransitionRewardModel
    
import common.math as m        # "m" 和官方 trainer 保持一致
from common import layers, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams
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
            self._dyn_vec = torch.compile(self.dynamics_vectorized,
                                          fullgraph=False,   # 允许小分支
                                          mode="reduce-overhead")
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
        
        self._pi = layers.mlp(self.latent_dim, 2*[self.hidden_dim], 2*self.action_dim)
        self._Qs = layers.Ensemble([layers.mlp(self.latent_dim + self.action_dim, 2*[self.hidden_dim], max(self.cfg.num_bins, 1), dropout=self.cfg.dropout) for _ in range(self.cfg.num_q)])
        self.apply(init.weight_init)
        init.zero_(self._Qs.params["2", "weight"])
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

    def encode(self, obs, task=None):
        # 1) 不是"真正的"批量轨迹，就当作单帧
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
        预测单步潜在状态
        
        Args:
            z: 当前潜在状态 [batch_size, latent_dim]
            a: 动作 [batch_size, action_dim]
            task: 任务ID（可选）
            
        Returns:
            next_z: 下一个潜在状态 [batch_size, latent_dim]
        """
        # 直接调用core，获取next_z
        next_z, _ = self.core(z, a, task)
        return next_z
    
    def reward(self, z, actions, task=None):
        """
        计算奖励序列，逐步调用core
        
        Args:
            z: [B, D] 初始潜在状态
            actions: [B, T, A] 动作序列或 [B, A] 单个动作
            task: 任务ID（可选）
        
        Returns:
            reward_seq: [B, T, reward_dim] 或 [B, reward_dim]
        """
        # 单步奖励预测
        if actions.dim() == 2:
            _, reward = self.core(z, actions, task)
            return reward  # [B, reward_dim]
        
        # 多步奖励预测 - 逐步推进
        B, T, A = actions.shape
        rewards = []
        
        # 逐步推进
        z_current = z.clone()  # [B, D]
        for t in range(T):
            next_z, r_t = self.core(z_current, actions[:, t], task)
            rewards.append(r_t)
            z_current = next_z  # 更新状态
        
        return torch.stack(rewards, dim=1)  # [B, T, reward_dim]

    def pi(self, z, task = None):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """

        # Gaussian policy prior
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = m.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        if self.cfg.multitask: # Mask out unused action dimensions
            mean = mean * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else: # No masking
            action_dims = None

        log_prob = m.gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1] if action_dims is None else action_dims
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = m.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict({
            "mean": mean,
            "log_std": log_std,
            "action_prob": 1.,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        })
        return action, info
    
    def Q(self, z, a, task=None, return_type='min', target=False, detach=False):
        """
        如果  detach=True:
            • 关闭梯度计算
            • 临时把 online-ensemble 切到 eval()
        """
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
        Q = m.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2
    
    
    def dynamics(self, z, actions, task=None):
        """
        多步潜在状态序列预测 - 状态完全串联
        
        Args:
            z: [B, D] 初始潜在状态
            actions: [B, T, A] 动作序列
            task: 任务ID（可选）
            
        Returns:
            z_seq: [B, T, D] 预测的潜在状态序列
        """
        B, T, A = actions.shape
        z_list = []
        
        # 单步推进状态
        z_current = z  # [B, D]
        for t in range(T):
            z_current = self.dynamics_step(z_current, actions[:, t], task)
            z_list.append(z_current)
        
        return torch.stack(z_list, dim=1)  # [B, T, D]

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def train(self, mode=True):
        """
        重写train方法以保持目标Q网络处于评估模式
        """
        super().train(mode)
        self._target_Qs.train(False)
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
        repr_str += f"Q-functions: Ensemble(n_models={len(self._Qs)})\n"
        
        return repr_str
        
            

