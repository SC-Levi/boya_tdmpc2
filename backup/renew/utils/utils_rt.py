import torch
from typing import Any


# ---------- shape helpers ----------
def normalize_bt(x: torch.Tensor, name: str) -> torch.Tensor:
    """Ensure tensor is [B,T,…].  Accept [B,D] or [B,T,D]."""
    if x.dim() == 2:
        return x[:, None]
    if x.dim() == 3:
        return x
    raise ValueError(f"{name} must be 2D or 3D, got {x.shape}")


# ---------- device ----------
def ensure_device(obj: Any, device: torch.device):
    if obj is None or not torch.is_tensor(obj):
        return obj
    return obj.to(device, non_blocking=True)


# ---------- env-to-cfg ----------
def update_cfg_from_env(cfg, env):
    cfg.obs_shape = {"state": env.observation_space.shape}
    cfg.obs_dim = env.observation_space.shape[0]
    cfg.action_dim = env.action_space.shape[0]
    return cfg


# ---------- AMP + clip ----------
def optim_step(optim, scaler, cfg):
    scaler.unscale_(optim)                       # ① 反-scale 梯度
    for group in optim.param_groups:             # ② 裁剪
        torch.nn.utils.clip_grad_norm_(group["params"], cfg.grad_clip_norm)
    scaler.step(optim)                           # ③ 更新参数
    scaler.update()                              # ④ **别忘了让 GradScaler 进入下一轮**
