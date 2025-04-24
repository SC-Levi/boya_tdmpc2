import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
from moore_tdmpc.layers import MooreExpertsModule


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
                 reward_dim: int = 1,
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
        
        # 将奖励区间 [-1, 1] 离散化为 reward_dim 个格子，并存储每个格子的中心值
        # 这将用于从离散化分布中计算期望奖励值
        if reward_dim > 1:
            self.register_buffer('reward_bin_centers', torch.linspace(-1, 1, reward_dim))
        else:
            self.register_buffer('reward_bin_centers', torch.tensor([0.0]))

        # 1) 共享 MoE 专家网络：输入为 (latent_dim + action_dim) → hidden_dim → latent_dim/reward_dim
        self.experts = MooreExpertsModule(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=latent_dim,    # experts 输出 latent 维度
            n_experts=n_experts
        )

        # 2) 预投影层，将 (z||a) → hidden_dim
        self.pre_proj = nn.Linear(latent_dim + action_dim, hidden_dim)

        # 3) 新代码 ——一个 Linear 输出 2E 维，后续再分片
        self._gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * n_experts)     # 一次出 2E 维
        )

        # 5) 融合 Head ：一次 matmul 同时得到 next_z & reward
        # dyn 输出维度 = latent_dim
        # rew 输出维度 = reward_dim
        self.head_W = nn.Parameter(torch.empty(          # [2, D, O_i]
            2,
            latent_dim,
            max(latent_dim, reward_dim)   # 兼容两种宽度
        ))
        self.head_b = nn.Parameter(torch.zeros(          # [2, O_i]
            2,
            max(latent_dim, reward_dim)
        ))
        nn.init.xavier_uniform_(self.head_W.reshape(2, -1))



        # 7) Value head：直接预测状态值函数
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        
    def dyn_head(self, z_feat, *_, **__):
        """
        只用融合后的 dyn_feat，动态从 head_W/head_b 切片，
        确保跟着 model.to(device) 一起搬。
        """
        W = self.head_W[0, :, :self.latent_dim]   # [D, D], 已在正确 device
        b = self.head_b[0, :self.latent_dim]      # [D]
        return z_feat @ W + b                     # [N, D]

    def reward_head(self, z_feat, *_, **__):
        """
        只用融合后的 rew_feat，动态从 head_W/head_b 切片，
        兼容离散(>1)和连续(=1)两种 reward_dim。
        """
        W = self.head_W[1, :, :self.reward_dim]   # [D, R]
        b = self.head_b[1, :self.reward_dim]      # [R]
        return z_feat @ W + b                     # [N, R]
    
    def forward(self, z: torch.Tensor, a: torch.Tensor, task_ids=None):
        """
        z:[N,D]  a:[N,A] → next_z, reward, (w_dyn, w_rew)
        """
        N = z.size(0)
        h = self.pre_proj(torch.cat((z, a), dim=-1))              # [N,H]

        gate_logits = self._gate(h)                               # [N,2E]
        logits_dyn, logits_rew = gate_logits.chunk(2, dim=-1)
        w_dyn = F.softmax(logits_dyn / self.temperature, dim=-1)
        w_rew = F.softmax(logits_rew / self.temperature, dim=-1)

        expert_feats = self.experts(h).permute(1, 0, 2)           # [N,E,D]
        dyn_feat = torch.einsum('ned,ne->nd', expert_feats, w_dyn)
        rew_feat = torch.einsum('ned,ne->nd', expert_feats, w_rew)

        # ⬇ 直接调“单参版” head
        next_z = self.dyn_head(dyn_feat).add_(z) 
        reward = self.reward_head(rew_feat)

        return next_z, reward, (w_dyn, w_rew)

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
        
    def core_vectorized(self, z_seq, a_seq, task_ids=None):
        """
        向量化处理整个序列，同时返回dynamics和reward预测
        
        Args:
            z_seq: 初始潜在状态 [B, latent_dim]
            a_seq: 动作序列 [B, T, action_dim] 或 [T, B, action_dim]
            task_ids: 可选的任务ID [B]
            
        Returns:
            next_z_seq: 下一个潜在状态序列 [B, T, latent_dim]
            reward_seq: 奖励序列 [B, T, reward_dim]
            (w_dyn_seq, w_rew_seq): 专家权重序列 ([B, T, n_experts], [B, T, n_experts])
        """
        # 确保z_seq是 [B, D]
        assert z_seq.dim() == 2, f"z_seq必须是2D [B, D]，当前为{z_seq.shape}"
        
        # 标准化a_seq为 [B, T, A] 格式
        if a_seq.dim() == 3:
            # 判断是 [B, T, A] 还是 [T, B, A]
            if a_seq.shape[0] > a_seq.shape[1] or a_seq.shape[0] == z_seq.shape[0]:
                # 已经是 [B, T, A]
                a_bt = a_seq
            else:
                # 格式是 [T, B, A]，转换为 [B, T, A]
                a_bt = a_seq.permute(1, 0, 2).contiguous()
            
            # 验证批次维度匹配
            assert z_seq.shape[0] == a_bt.shape[0], (
                f"批次维度不匹配: z_seq={z_seq.shape[0]}, a_bt={a_bt.shape[0]}"
            )
        else:
            # 处理2D动作，添加时间维度: [B, A] -> [B, 1, A]
            a_bt = a_seq.unsqueeze(1)
        
        # 获取维度
        batch_size, seq_len, action_dim = a_bt.shape
        latent_dim = z_seq.shape[1]
        
        # 自回归计算序列
        # 创建一个临时张量存储每个时间步的状态
        z_states = torch.zeros(batch_size, seq_len, latent_dim, device=z_seq.device)
        z_states[:, 0] = z_seq  # 第一个状态是输入的初始状态
        
        # 计算所有奖励和专家权重结果
        reward_seq = torch.zeros(batch_size, seq_len, self.reward_dim, device=z_seq.device)
        w_dyn_seq = torch.zeros(batch_size, seq_len, self.n_experts, device=z_seq.device)
        w_rew_seq = torch.zeros(batch_size, seq_len, self.n_experts, device=z_seq.device)
        
        # 自回归计算序列
        for t in range(seq_len):
            # 获取当前状态和动作
            z_t = z_states[:, t]  # [B, latent_dim]
            a_t = a_bt[:, t]      # [B, action_dim]
            
            # 只有第一个时间步使用原始z_seq，其余步骤使用上一步的预测
            if t < seq_len - 1:
                # 计算下一个状态，并存储
                z_next, reward, (w_dyn, w_rew) = self(z_t, a_t, task_ids)
                z_states[:, t+1] = z_next
            else:
                # 最后一步只需计算reward和权重
                _, reward, (w_dyn, w_rew) = self(z_t, a_t, task_ids)
            
            # 存储每个时间步的结果
            reward_seq[:, t] = reward
            w_dyn_seq[:, t] = w_dyn
            w_rew_seq[:, t] = w_rew
        
        # 最终结果：z_states 包含全部时间步的状态，含初始输入
        next_z_seq = z_states  # [B, T, latent_dim]
        
        return next_z_seq, reward_seq, (w_dyn_seq, w_rew_seq) 
 