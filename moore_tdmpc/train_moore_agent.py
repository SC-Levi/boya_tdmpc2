#!/usr/bin/env python
# 禁用 Kineto，避免所有 RecordFunction 的内部断言
import os
os.environ["KINETO_ENABLED"] = "0"

import torch
import numpy as np
import torch.nn.functional as F
import sys
from typing import Dict, List, Optional, Tuple, Union, Any

# Import paths to access modules from TDMPC2 repo
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# TDMPC2 imports
from TDMPC2.algorithms.tdmpc2 import TDMPC2
from TDMPC2.algorithms.envs import make_env
from TDMPC2.algorithms.common.buffer import Buffer
from TDMPC2.algorithms.common.logger import Logger
from TDMPC2.algorithms.trainer.offline_trainer import OfflineTrainer
from TDMPC2.algorithms.trainer.online_trainer import OnlineTrainer
from TDMPC2.algorithms.common.seed import set_seed

# Moore-specific imports
from moore_tdmpc.world_model import MooreWorldModel
from moore_tdmpc.task_encoder import MooreTaskEncoder

# Configure environment
os.environ['MUJOCO_GL'] = "osmesa"
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"

# Configure PyTorch backend
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

import torch._dynamo as dynamo
dynamo.reset() 
dynamo.config.capture_scalar_outputs = True

# Tensor normalization utility functions
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

def ensure_device(x, device):
    """
    Ensure tensor is on the specified device.
    
    Args:
        x: A tensor or a collection of tensors (list, tuple, dict)
        device: Target device 
        
    Returns:
        The tensor or collection on the target device
    """
    if x is None:
        return None
        
    if isinstance(x, torch.Tensor):
        # Only move if not already on correct device to avoid unnecessary transfers
        if x.device != device:
            return x.to(device)
        return x
    elif isinstance(x, (list, tuple)):
        return type(x)(ensure_device(item, device) for item in x)
    elif isinstance(x, dict):
        return {k: ensure_device(v, device) for k, v in x.items()}
    else:
        # Non-tensor types are returned as is
        return x

def update_cfg_from_env(cfg, env):
    """
    Update configuration with properties from the environment
    
    Args:
        cfg: Configuration object
        env: Environment object from make_env
        
    Returns:
        Updated configuration object
    """
    if hasattr(env, 'observation_space'):
        if hasattr(env.observation_space, 'spaces'):
            setattr(cfg, 'obs_shape', {k: v.shape for k, v in env.observation_space.spaces.items()})
        else:
            obs_key = getattr(cfg, 'obs', 'state')
            setattr(cfg, 'obs_shape', {obs_key: env.observation_space.shape})
                
        if hasattr(env.observation_space, 'shape'):
            setattr(cfg, 'obs_dim', env.observation_space.shape[0])
    
    if hasattr(env, 'action_space') and hasattr(env.action_space, 'shape'):
        env_action_dim = env.action_space.shape[0]
        setattr(cfg, 'env_action_dim', env_action_dim)
        
    if not hasattr(cfg, 'action_dim'):
            setattr(cfg, 'action_dim', env_action_dim)
    
    return cfg

@torch._dynamo.disable()    
def optim_step(model, scaler, cfg):
    scaler.unscale_(model.optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
    scaler.step(model.optim)
    scaler.update()


class MooreTDMPC2(TDMPC2):
    def __init__(self, 
                 cfg, 
                 env=None,
                 deterministic=False, 
                 model_dir=None, 
                 demo_mode=False,
                 load_episode=None,
                 gpt_model=None,
                 debug_prints=False):
        """Initialize the MooreTDMPC2 agent."""


        cfg.compile = False          # ← 新增一行
        super().__init__(cfg)        # 之后照常
        # 先给 buffer 占个位
        self.buffer = None
        
        # Store the core configuration and environment
        self.cfg = cfg
        self.env = env
        
        # Store other parameters
        self.deterministic = deterministic
        self.model_dir = model_dir
        self.demo_mode = demo_mode
        self.load_episode = load_episode
        self.gpt_model = gpt_model
        self.debug_prints = debug_prints
        
        # Explicitly set device and handle device override if provided
        if hasattr(self.cfg, 'disable_cuda') and self.cfg.disable_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        self.cfg.device = self.device  # Add device to config for consistency
        
        # Initialize gradient scaler for mixed precision training
        use_amp = getattr(self.cfg, 'use_mixed_precision', False) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler(enabled=False)
        
        # 默认开启梯度裁剪，防止梯度爆炸导致 NaN
        self.cfg.grad_clip_norm = getattr(self.cfg, 'grad_clip_norm', 5.0)
        
        # Ensure necessary Moore-specific parameters are set
        if hasattr(self.cfg, 'moore'):
            if isinstance(self.cfg.moore, dict):
                moore_n_experts = self.cfg.moore.get('n_experts', 4)
                moore_temperature = self.cfg.moore.get('temperature', 1.0)
                moore_use_softmax = self.cfg.moore.get('use_softmax', True)
                moore_debug_task_emb = self.cfg.moore.get('debug_task_emb', False)
            else:
                moore_n_experts = getattr(self.cfg.moore, 'n_experts', 4)
                moore_temperature = getattr(self.cfg.moore, 'temperature', 1.0)
                moore_use_softmax = getattr(self.cfg.moore, 'use_softmax', True)
                moore_debug_task_emb = getattr(self.cfg.moore, 'debug_task_emb', False)
                
            # Set parameters directly on cfg
            self.cfg.n_experts = moore_n_experts
            self.cfg.moore_temperature = moore_temperature  # Store original Moore temperature
            self.cfg.use_softmax = moore_use_softmax
            self.cfg.debug_task_emb = moore_debug_task_emb
            self.cfg.debug_mode = moore_debug_task_emb
        else:
            # Default values if no Moore-specific config
            self.cfg.n_experts = getattr(self.cfg, 'n_experts', 4)
            self.cfg.moore_temperature = getattr(self.cfg, 'moore_temperature', 1.0)
            self.cfg.use_softmax = getattr(self.cfg, 'use_softmax', True)
            self.cfg.debug_task_emb = getattr(self.cfg, 'debug_task_emb', False)
            self.cfg.debug_mode = getattr(self.cfg, 'debug_task_emb', False)
        
        # Temperature used for dynamics MoE
        self.cfg.temperature = getattr(self.cfg, 'temperature', 0.5)  # Default from moore_config.yaml
        
        # Required dimensions for Moore models
        self.cfg.expert_hidden_dims = getattr(self.cfg, 'expert_hidden_dims', [256, 256])
        self.cfg.task_dim = getattr(self.cfg, 'task_dim', 96)
        
        # Set encoder and MoE hidden dimensions if not present
        self.cfg.encoder_hidden_dim = getattr(self.cfg, 'encoder_hidden_dim', 256)
        self.cfg.moe_hidden_dim = getattr(self.cfg, 'moe_hidden_dim', 256)
        
        # Ensure obs_shape is set
        if not hasattr(self.cfg, 'obs_shape'):
            obs_dim = getattr(self.cfg, 'obs_dim', 24)
            setattr(self.cfg, 'obs_shape', {'state': (obs_dim,)})
        
        # Update config from environment if provided
        if env is not None:
            self.cfg = update_cfg_from_env(self.cfg, env)
        
        # Call the parent constructor with the updated config
        try:
            super().__init__(self.cfg)
        except Exception as e:
            raise
        
        # Create a Moore world model to replace the standard one
        self.model = self._create_world_model(self.cfg)
        
        # Update optimizer to ensure it targets the right parameters
        self.model.optim = torch.optim.Adam([
            {'params': self.model.task_encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model.core.parameters()},
            {'params': self.model._pi.parameters()},
            {'params': self.model._q_networks.parameters()}
        ], lr=self.cfg.lr)
        
        # Initialize the policy optimizer 
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(), 
            lr=self.cfg.lr
        )

        if getattr(cfg, 'compile_loss', False):
            self._compiled_loss_and_backward = torch.compile(self._loss_and_backward,
                                                            mode='reduce-overhead')
        else:
            self._compiled_loss_and_backward = self._loss_and_backward

    
    def _create_world_model(self, cfg):
        """
        Create a Moore world model that implements the mixture of experts architecture,
        but patch encode() so it always returns only the latent Tensor.
        """
        # Ensure environment dimensions are properly set
        if not hasattr(cfg, 'obs_dim'):
            cfg.obs_dim = 24
            
        if not hasattr(cfg, 'action_dim'):
            cfg.action_dim = 24
        
        # Create and return the Moore world model
        model = MooreWorldModel(cfg).to(self.device)
            
        # —— 开始补丁 —— 
        orig_encode = model.encode
        def encode_only(obs, task=None):
            # (1)  time-first → batch-first
            if obs.dim() == 3 and obs.shape[0] < obs.shape[1]:
                obs = obs.permute(1, 0, 2).contiguous()

            # ---------- 这里只在 task 存在时再管它 ----------
            if task is not None:
                # time-first [T,B,1] → [B,T,1]
                if task.dim() == 3 and task.shape[2] == 1:
                    task = task.permute(1, 0, 2).contiguous()

                # [B,T] / [B,1] → [B]
                if task.dim() == 3 and task.shape[2] == 1:
                    task = task[:, 0, 0]          # [B]
                elif task.dim() == 2:
                    task = task[:, 0]             # [B]

            # (2) 兼容 2-D 扁平输入
            squeezed = False
            if obs.dim() == 2:                    # [N,D]
                obs = obs.unsqueeze(1)            # [N,1,D]
                if task is not None and task.dim() == 1:
                    task = task.unsqueeze(1)      # [N,1]
                squeezed = True

            # (3) 调用原始 encode
            z = orig_encode(obs, task) if task is not None else orig_encode(obs)
            z = z[0] if isinstance(z, tuple) else z
            if squeezed:
                z = z.squeeze(1)
            return z


        model.encode = encode_only
        # —— 补丁结束 —— 
        
        return model
    

    @torch._dynamo.disable() 
    def _update(self, obs, action, reward, task=None):
        """Update model parameters from replay buffer samples.
        
        Args:
            obs: Observations [batch_size, horizon+1, obs_dim]
            action: Actions [batch_size, horizon, action_dim]
            reward: Rewards [batch_size, horizon, 1]
            task: Task identifier (optional)
            
        Returns:
            Dictionary of statistics
        """
        
        # Device checks
        device = self.model.device
        obs     = ensure_device(obs,     device)
        action  = ensure_device(action,  device)
        reward  = ensure_device(reward,  device)
        task    = ensure_device(task,    device) if task is not None else None

        B, Tp1, _ = obs.shape        # Tp1 == H+1
        H = Tp1 - 1

        # 2‧ 前向：encode + TD-target
        with torch.no_grad():
            # latent_seq: [B , T+1 , D]
            z_seq = normalize_bt(self.model.encode(obs, task), 'obs')
            if z_seq.shape[0] == Tp1 and z_seq.shape[1] == B:
                z_seq = z_seq.permute(1, 0, 2).contiguous()

            if task is not None and getattr(self, "has_task_encoder", False):
                task_id = task.squeeze(-1) if task.dim() == 2 else task       # [B]
                z_seq_task = torch.cat(
                    [z_seq,
                    self.model.task_encoder(task_id).unsqueeze(1).repeat(1, Tp1, 1)],
                    dim=-1
                )
            else:
                z_seq_task = z_seq

            H_act  = action.shape[1]          # = cfg.horizon
            z      = z_seq_task[:, :H_act]    # [B , H , D]
            z_next = z_seq_task[:, 1:H_act+1] # [B , H , D]

            B, _, D = z.shape
            z_flat       = z      .reshape(-1, D) # [B*H , D]
            z_next_flat  = z_next .reshape(-1, D)

            current_value = self.model.core.value_head(z_flat)      # [B*H,1]
            next_value = self.model.core.value_head(z_next_flat) # [B*H,1]

            current_value = current_value.view(B, H, 1)
            next_value = next_value.view(B, H, 1)
            td_target = self._td_target(next_value, reward, gamma=self.cfg.gamma)

        pl, rl, vl, cl, tot = self._compiled_loss_and_backward(
            z_seq_full=z_seq,      # 直接传 B×H×D
            action=action,
            reward=reward,
            task=task,
            td_target=td_target
        )

        optim_step(self.model, self.scaler, self.cfg)

        stats = {
            "policy_loss":      pl.item(),
            "reward_loss":      rl.item(),
            "value_loss":       vl.item(),
            "consistency_loss": cl.item(),
            "total_loss":       tot.item(),
        }
        return stats
    
    def _loss_and_backward(self, z_seq_full, action, reward, task, td_target):
        reward_dim = self.model.core.reward_dim
        B, H, _ = action.shape
        Tp1, D  = z_seq_full.shape[1:3]
        assert Tp1 == H + 1, f"z_seq 时序维度应该是 action 时序＋1，实际是 {Tp1} vs {H+1}"

        # ----------- 展平 -----------
        z_flat = z_seq_full[:, :-1].reshape(-1, D)                  # [B*H,D]
        a_flat = action.reshape(-1, self.model.action_dim)     # [B*H,A]
        r_flat = reward.reshape(-1)                            # [B*H]
        td_flat = td_target.reshape(-1, 1)                     # [B*H,1]

        # ===========================  forward & loss  ===========================
        with torch.cuda.amp.autocast(enabled=self.cfg.use_mixed_precision):
            # 1) policy loss  (Linear 层在 autocast 下会自动把权重转换为 FP16)
            mean, std  = self.model.policy(z_flat)             # [B*H,A]
            dist       = torch.distributions.Normal(mean, std)
            policy_loss = -dist.log_prob(a_flat).sum(-1).mean()

            # 2) value loss  （value_head 是 nn.Linear，同样安全）
            q_flat     = self.model.core.value_head(z_flat)    # [B*H,1]
            value_loss = F.mse_loss(q_flat.view(B, H, 1), td_target)

        # 3) reward loss —— **禁用 autocast，强制在 FP32 下跑一遍 core**
        #    这样就不会再出现 Half vs Float 的 addmm 冲突
        with torch.cuda.amp.autocast(enabled=False):
            _, rew_flat, _ = self.model.core(z_flat.float(), a_flat.float())
            if reward_dim > 1:
                reward_loss = F.cross_entropy(rew_flat, r_flat.long())
            else:
                reward_loss = F.mse_loss(rew_flat.view(B, H, 1), reward.float())

        # 4) 其它损失（可选）
        consistency_loss = torch.tensor(0., device=z_flat.device)
        total_loss = (
            self.cfg.entropy_coef   * policy_loss +
            self.cfg.value_coef     * value_loss  +
            self.cfg.reward_coef    * reward_loss +
            self.cfg.consistency_coef * consistency_loss
        ).nan_to_num()          # 避免 NaN

        # ===========================  backward  ===========================
        total_loss.backward()
        return policy_loss, reward_loss, value_loss, consistency_loss, total_loss



    
    def step(self, z, a, task=None):
        """
        Performs a single step in the dynamics model and returns policy outputs.
        
        Args:
            z: Latent state [batch_size, latent_dim]
            a: Action [batch_size, action_dim]
            task: Task identifier (optional)
            
        Returns:
            next_z: Next latent state [batch_size, latent_dim]
            mu: Action mean [batch_size, action_dim]
            log_std: Action log std [batch_size, action_dim]
            log_pi: Action log probability [batch_size]
        """
        # Step the dynamics model to get the next latent state using the core model
        next_z, _, _ = self.model.core(z, a, task)
        # Get policy outputs from the next state
        action, info = self.model.pi(next_z, task)
        
        # Extract policy statistics
        mu = info["mean"]
        log_std = info["log_std"]
        log_pi = -info["entropy"]  # Negative entropy is log probability
        
        return next_z, mu, log_std, log_pi
    
    def _td_target(self, next_v_c, reward, gamma=None):
        """
        next_v_c : [B,H] 或 [B,H,1] 或 [B,H,R]  
        reward   : [B,H,1]
        返回     : [B,H,1]
        """
        if gamma is None:
            gamma = self.cfg.gamma

        r = reward.squeeze(-1)                  # → [B,H]
        # next_v_c 可能 2D 也可能 3D
        if next_v_c.dim() == 3:
            v = next_v_c.mean(-1)               # [B,H]
        else:
            v = next_v_c                        # [B,H]

        assert r.shape == v.shape, \
            f"TD-target shape mismatch: reward {r.shape}, value {v.shape}"

        td = r + gamma * v                      # [B,H]
        return td.unsqueeze(-1)                 # → [B,H,1]

    
    def update(self, buffer, *args, **kwargs):
        obs, action, reward, *rest = buffer.sample()
        task = rest[0] if rest else None
        # ------------------------------------------------------------------
        #  1) 统一 action 变成 batch-first [B,H,A]
        # ------------------------------------------------------------------
        assert action.dim() == 3, "action 必须 3-D"

        H_cfg = self.cfg.horizon                       # e.g. 5
        if action.shape[0] == H_cfg:                   # time-first
            action = action.permute(1, 0, 2).contiguous()
        elif action.shape[1] == H_cfg:                 # already batch-first
            pass
        else:
            raise ValueError(f"action 形状不符合预期 {action.shape}, horizon={H_cfg}")

        B, H, A = action.shape        # 现在 H == H_cfg

        # ------------------------------------------------------------------
        #  2) 统一 obs 变成 batch-first [B,H+1,D]
        # ------------------------------------------------------------------
        assert obs.dim() == 3, "obs 必须 3-D"
        if obs.shape[0] == B and obs.shape[1] == H+1:      # batch-first OK
            pass
        elif obs.shape[0] == H+1 and obs.shape[1] == B:    # time-first
            obs = obs.permute(1, 0, 2).contiguous()
        elif obs.shape[0] == B and obs.shape[1] == H:      # 少最后一帧
            obs = torch.cat([obs, obs[:, -1:, :].clone()], dim=1)
        else:
            raise ValueError(f"无法识别的 obs 形状 {obs.shape}")

        # ————————————————
        # 3) 统一 reward 变成 [B, H, 1]
        if reward.dim() == 3:
            # time-first: [H, B, 1] -> [B, H, 1]
            if reward.shape[0] == H and reward.shape[1] == B:
                reward = reward.permute(1, 0, 2).contiguous()
            # batch-first 完整: [B, H, 1]
            elif reward.shape[0] == B and reward.shape[1] == H:
                pass
            else:
                raise ValueError(f"无法识别的 reward 形状 {reward.shape}")
        elif reward.dim() == 2:
            # 如果是 [B, H]，加一个通道维
            if reward.shape[0] == B and reward.shape[1] == H:
                reward = reward.unsqueeze(-1)
            else:
                raise ValueError(f"无法识别的 reward 2D 形状 {reward.shape}")
        else:
            raise ValueError(f"reward 必须是 2D 或 3D，当前 {reward.dim()}D")

        if task is not None:
            if task.dim() == 2:                         # 可能是 [H,B] 或 [B,H]
                # time-first [H,B] -> [B,H]
                if task.shape[0] == H and task.shape[1] == B:
                    task = task.permute(1, 0).contiguous()
                # 现在 task 是 [B,H]；取第 0 步即可
                task = task[:, 0]                       # -> [B]
            elif task.dim() != 1:
                raise ValueError(f"无法识别的 task 形状 {task.shape}")
            # 最终扩成 [B,1]
            task = task.unsqueeze(-1)                   # -> [B,1]

        return self._update(obs, action, reward, task=task)




def dict_to_object(d):
    """
    Convert a dictionary to an object with attributes and dict-like access.
    
    Args:
        d: Dictionary to convert
        
    Returns:
        object: Object with attributes and dict-like access from the dictionary
    """
    class Config:
        def __init__(self, data=None):
            if data is None:
                data = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    value = Config(value)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            value[i] = Config(item)
                setattr(self, key, value)
                
        def get(self, key, default=None):
            return getattr(self, key, default)
            
        def __getitem__(self, key):
            return getattr(self, key)
            
        def __setitem__(self, key, value):
            setattr(self, key, value)
            
        def __contains__(self, key):
            return hasattr(self, key)
            
        def __repr__(self):
            attrs = ', '.join(f"{key}={repr(value)}" for key, value in self.__dict__.items())
            return f"Config({attrs})"
    
    if isinstance(d, dict):
        return Config(d)
    return d



