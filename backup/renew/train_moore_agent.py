#!/usr/bin/env python
import os
import torch
import numpy as np
from common.math import two_hot, two_hot_inv, gumbel_softmax_sample
import torch.nn.functional as F
# TDMPC2 imports
from common.scale import RunningScale
from utils.utils_rt import (
    update_cfg_from_env,
)
from tensordict import TensorDict

# Moore-specific imports
from common.world_model import MooreWorldModel

# Configure PyTorch backend
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


class MooreTDMPC2:
    def __init__(
        self,
        cfg,
        env=None,
    ):
        """Initialize the MooreTDMPC2 agent."""
        # Initialize buffer placeholder
        self.buffer = None

        # Store configuration and environment
        self.cfg = cfg
        self.device = "cuda:0"
        self.env = env
        # Initialize gradient scaler for mixed precision training
        use_amp = (
            getattr(self.cfg, "use_mixed_precision", False)
        )
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))


        # Update config from environment if provided
        if env is not None:
            self.cfg = update_cfg_from_env(self.cfg, env)

        # Create Moore world model
        self.model = MooreWorldModel(cfg).to(self.device)

        # Setup optimizers
        self.optim = torch.optim.Adam([
            {"params": self.model.task_encoder.parameters(), "lr": self.cfg.lr * self.cfg.enc_lr_scale},
            {"params": self.model.core.parameters()},
            {"params": self.model._Qs.parameters()}
        ], lr=self.cfg.lr, capturable=True)
        self.scale = RunningScale(self.cfg).to(self.device)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)

        # 在创建模型和优化器后添加这段代码
        if hasattr(self.cfg, "multitask") and self.cfg.multitask:
            self.discount = torch.tensor(
                [self._get_discount(ep_len) for ep_len in self.cfg.episode_lengths], 
                device=self.device
            )
        else:
            self.discount = self._get_discount(self.cfg.episode_length)

    def _ensure_task_id(self, task):
        """确保在多任务模式下有有效的任务ID"""
        if hasattr(self.cfg, "multitask") and self.cfg.multitask:
            if task is None:
                return torch.zeros(1, dtype=torch.long, device=self.device)
            elif not isinstance(task, torch.Tensor):
                return torch.tensor([task], dtype=torch.long, device=self.device)
            elif task.dim() == 0:  # 是0维tensor，需要扩展
                return task.unsqueeze(0)
            return task
        return task  # 非多任务模式，直接返回原值

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """计算TD目标，与原版tdmpc2保持一致"""
        task = self._ensure_task_id(task)    
        action, _ = self.model.pi(next_z, task)
        
        # 采用与原版一致的方式处理折扣
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        
        # 确保Q值具有正确的维度
        q_val = self.model.Q(next_z, action, task, return_type='min', target=True)
        
        # 确保维度匹配，原版写法直接广播
        return reward + discount * q_val

    
    @torch.no_grad()
    def plan(self, obs: torch.Tensor, t0: bool = False, eval_mode: bool = False, task=None) -> torch.Tensor:
        """
        Model-predictive control planning using MPPI-style CEM.
        输入:
            obs       (shape=(obs_dim,) or (1,obs_dim))  单个时刻观测
            t0        是否为回合首帧，用于 warm-start
            eval_mode 是否只取均值
            task      (optional) 任务索引
        返回:
            action (shape=(action_dim,))
        """
        # 确保在多任务模式下有有效的任务ID
        task = self._ensure_task_id(task)
        
        # 1) encode 初始潜状态
        z0 = self.model.encode(obs.to(self.device).unsqueeze(0), task)  # (1,D)
        H, N, M, E = self.cfg.horizon, self.cfg.num_samples, self.cfg.num_pi_trajs, self.cfg.num_elites
        A = self.cfg.action_dim
        device = self.device

        # 2) policy warm-start 的 M 条轨迹
        if M > 0:
            pi_actions = torch.empty(H, M, A, device=device)
            z_pi = z0.repeat(M, 1)  # [M, D]
            
            for t in range(H):
                # 获取策略动作
                pi_actions[t], _ = self.model.pi(z_pi, task)  # [M, A]
                # 使用dynamics_step直接更新状态
                z_pi = self.model.dynamics_step(z_pi, pi_actions[t], task)
        # 3) 初始化 CEM 分布
        mean = torch.zeros(H, A, device=device)
        std  = torch.full((H, A), self.cfg.max_std, device=device)
        if not t0:
            # warm-start 上一步的 mean
            mean[:-1] = self._prev_mean[1:]
        # 4) 分配候选动作张量
        actions = torch.empty(H, N, A, device=device)
        if M > 0:
            actions[:, :M] = pi_actions

        # 5) 重复初态以用于 rollout
        z_expand = z0.repeat(N, 1)  # (N, D)

        # 6) CEM / MPPI 循环
        for _ in range(self.cfg.iterations):
            # 6.1) 在剩余样本上采噪声
            eps = torch.randn(H, N - M, A, device=device)
            cand = mean.unsqueeze(1) + std.unsqueeze(1) * eps    # (H, N-M, A)
            cand = cand.clamp(-1, 1)
            actions[:, M:] = cand

            # 多任务下屏蔽动作维度
            if self.cfg.multitask:
                mask = self.model._action_masks[task]             # (A,)
                actions = actions * mask.view(1, 1, A)

            # 6.2) 评估所有候选
            #    z_expand 固定为 N 条从 z0 展开的轨迹起点
            values = self._estimate_value(z_expand, actions, task).nan_to_num(0)  # (N,1)
            values = values.squeeze(-1)                              # (N,)

            # 6.3) 选 Top-E elites
            topk_vals, topk_idx = torch.topk(values, E, dim=0)      # (E,)
            elite_actions = actions[:, topk_idx]                     # (H, E, A)
            elite_vals    = topk_vals                                # (E,)

            # 6.4) 计算权重（MPPI 公式）
            vmax = elite_vals.max(0, keepdim=True).values           # 标准化常数
            weights = torch.exp(self.cfg.temperature * (elite_vals - vmax))  # (E,)
            weights = weights / (weights.sum(0, keepdim=True) + 1e-9)         # (E,)

            # 6.5) 用加权精英样本更新 mean/std
            w_bc = weights.view(1, E, 1)  # (1,E,1) 方便广播
            mean = (elite_actions * w_bc).sum(dim=1) / (weights.sum() + 1e-9)  # (H,A)
            var  = ((elite_actions - mean.unsqueeze(1))**2 * w_bc).sum(dim=1) / (weights.sum() + 1e-9)
            std  = var.sqrt().clamp(self.cfg.min_std, self.cfg.max_std)       # (H,A)

            # 多任务下屏蔽
            if self.cfg.multitask:
                std = std * mask.view(1, A)
                mean = mean * mask.view(1, A)

        # 7) 从精英集合中随机挑一条，或 eval_mode 下直接取均值
        if eval_mode:
            best = mean[0]  # 取首步 mean
        else:
            # Gumbel-softmax 从 weights 中采样一个索引
            idx = gumbel_softmax_sample(weights).long()  # (E,)
            a_seq = elite_actions[:, idx]                     # (H,A)
            # 对首步加随机扰动
            best = a_seq[0] + std[0] * torch.randn(A, device=device)

        # 8) 保存下次 warm-start 用的 mean，返回动作
        self._prev_mean.copy_(mean)
        return best.clamp(-1, 1).cpu()


    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        选动作：要么用 MPC plan，要么调用 model.pi 取一步 policy。
        返回一个 1D Tensor, shape = [action_dim]
        """
        self.model.eval()
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)   # (1, obs_dim)
        
        # 确保在多任务模式下有有效的任务ID
        task = self._ensure_task_id(task)

        if self.cfg.mpc:
            # plan 已经返回 shape=[A]
            action = self.plan(obs, t0, eval_mode, task)
        else:
            # 用 model.pi 一步输出
            # 因为 pi 接受 shape=(batch,latent_dim) 或 shape=(1,latent_dim)
            action_seq, info = self.model.pi(obs, task)
            # action_seq 的 shape 可能是 (1,1,A) 或 (1,A)，取成 [A]
            # 这是最保险的写法：
            action = action_seq.view(-1, self.cfg.action_dim)[0]

            # 如果你想 eval_mode 下用 mean：
            if eval_mode:
                mean = info["mean"].view(-1, self.cfg.action_dim)
                action = mean[0]

        action = action.clamp(-1, 1)   # 限幅
        self.model.train()
        return action.cpu()

    def _update(self, obs, action, reward, task=None):
        """更新方法，增加一致性损失截断"""
        device = self.device
        obs, action, reward = obs.to(device), action.to(device), reward.to(device)
        task = self._ensure_task_id(task)
        
        B, T, A = action.shape
        
        # 计算目标
        with torch.no_grad():
            next_z = self.model.encode(obs[:, 1:], task)
            td_targets = self._td_target(next_z, reward, task)
        
        # 准备更新
        self.model.train()
        
        # 潜在空间回滚
        z_seq = torch.empty(T+1, B, self.model.latent_dim, device=device)
        z = self.model.encode(obs[:, 0], task)
        z_seq[0] = z
        
        # 计算一致性损失
        consistency_loss = 0
        for t in range(T):
            z = self.model.dynamics_step(z, action[:, t], task)
            consistency_loss += F.mse_loss(z, next_z[:, t]) * (self.cfg.rho**t)
            z_seq[t+1] = z
        
        # 平均一致性损失并应用阈值限制
        c_loss = consistency_loss / T
        
        # 添加一致性损失截断
        consistency_clip = getattr(self.cfg, "consistency_clip", 10.0)  # 默认阈值为10
        if c_loss > consistency_clip:
            print(f"Warning: Consistency loss {c_loss:.4f} exceeds threshold, clipping to {consistency_clip}")
            c_loss = torch.tensor(consistency_clip, device=device)
        
        # 计算值函数和奖励损失
        z_seq_except_last = z_seq[:-1]  # [T,B,D]
        
        # 计算值损失和奖励损失
        value_loss = 0.0
        reward_loss = 0.0
        
        for t in range(T):
            z_t = z_seq_except_last[t]  # [B,D]
            a_t = action[:, t]          # [B,A]
            r_t = reward[:, t]          # [B,1]
            td_t = td_targets[:, t]     # [B,1]
            
            # 计算值函数损失
            q_logits_t = self.model.Q(z_t, a_t, task, return_type='all')
            td_bins_t = two_hot(td_t, self.cfg)
            td_bins_t = td_bins_t.unsqueeze(0).expand(q_logits_t.size(0), -1, -1)
            value_loss_t = -(td_bins_t * F.log_softmax(q_logits_t, -1)).sum(-1).mean()
            value_loss += value_loss_t * (self.cfg.rho**t)
            
            # 计算奖励损失
            _, rew_logits_t = self.model.core(z_t, a_t, task)
            r_bins_t = two_hot(r_t, self.cfg)
            reward_loss_t = -(r_bins_t * F.log_softmax(rew_logits_t, -1)).sum(-1).mean()
            reward_loss += reward_loss_t * (self.cfg.rho**t)
        
        # 平均损失
        value_loss = value_loss / T
        reward_loss = reward_loss / T
        
        # 组合总损失
        total_loss = (
            self.cfg.consistency_coef * c_loss +  # 使用裁剪后的一致性损失
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss
        )
        
        # 更新模型
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)
        
        # 更新策略和目标Q网络
        pi_info = self.update_pi(z_seq.transpose(0, 1)[:, :-1].detach(), task)  # [B,T,D]
        self.model.soft_update_target_Q()
        self.model.eval()
        
        # 返回训练统计
        out = TensorDict({
            "consistency_loss": consistency_loss,
            "clipped_consistency_loss": c_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
        })
        out.update(pi_info)
        return out.detach().mean()

    @torch.no_grad()
    def _estimate_value(self, z0, actions, task=None):
        """遵循原版tdmpc2中的_estimate_value实现"""
        task = self._ensure_task_id(task)
        
        # 处理actions维度
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if actions.dim() == 3 and actions.shape[0] != z0.size(0):
            actions = actions.permute(1, 0, 2).contiguous()
        
        # 初始化累积奖励和折扣因子
        G, discount = 0, 1
        z = z0.clone()
        
        # 按原版方式循环
        for t in range(actions.shape[1]):
            # 获取奖励并处理
            r_t = self.model.reward(z, actions[:, t], task)
            if r_t.dim() > 1 and r_t.size(-1) > 1:
                reward = two_hot_inv(r_t, self.cfg)
            else:
                reward = r_t.squeeze(-1)
            
            # 更新状态
            z = self.model.dynamics_step(z, actions[:, t], task)
            
            # 累积折扣奖励
            G = G + discount * reward
            
            # 更新折扣 - 精确复制原版代码逻辑
            discount_update = self.discount[task] if self.cfg.multitask else self.discount
            discount = discount * discount_update
        
        # 添加终端值
        action, _ = self.model.pi(z, task)
        q_term = self.model.Q(z, action, task, return_type='avg')
        
        # 返回总估值
        return G + discount * q_term

    def update_pi(self, zs, task=None):
        """与原版tdmpc2保持一致的策略更新"""
        task = self._ensure_task_id(task)
        
        # 处理输入是时间-批次-特征格式[T,B,D]的情况
        if zs.dim() == 3 and zs.shape[0] > zs.shape[1]:
            # 时间优先格式，需要转置
            zs = zs.transpose(0, 1)  # [B,T,D]
        
        # 保持原版调用
        action, info = self.model.pi(zs, task)
        qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)
        
        # 计算策略损失
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
        
        # 反向传播和优化
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)
        
        # 返回训练统计信息
        return TensorDict({
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            "pi_scale": self.scale.value,
        })

    def update(self, buffer):
        obs, action, reward, task = buffer.sample()  # 全都是 Batch-first

        B, T, _ = obs.shape[0], self.cfg.horizon, obs.shape[2]
        assert action.shape == (B, T, self.cfg.action_dim)
        assert reward.shape in [(B, T), (B, T, 1)]
        if reward.dim()==2: reward = reward.unsqueeze(-1)
        if task is not None:
            if task.dim()==2 and task.size(1)==1:
                task = task.squeeze(-1)

        return self._update(obs, action, reward, task=task)


    def _get_discount(self, episode_length):
        """
        根据回合长度计算折扣因子。
        使用线性缩放的启发式方法。
        
        Args:
            episode_length (int): 回合长度。假设回合是固定长度的。
            
        Returns:
            float: 任务的折扣因子。
        """
        frac = episode_length / self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)


__all__ = ["MooreTDMPC2"]
