# ======= eval() 首次剖析 =========
import cProfile, pstats, io
import sys, os
# 确保正确导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TDMPC2.algorithms.trainer.online_trainer import OnlineTrainer

# 保存原始 eval
_original_eval = OnlineTrainer.eval

def profiled_eval(self, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    # 真正执行 eval
    result = _original_eval(self, *args, **kwargs)
    pr.disable()
    # 打印前 20 条累积时间最多的函数
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    stats.print_stats(20)
    print("\n=== 首次 eval() 阶段热点函数 ===")
    print(s.getvalue())
    # 恢复原始方法，避免重复分析
    OnlineTrainer.eval = _original_eval
    return result

OnlineTrainer.eval = profiled_eval
# =================================

#!/usr/bin/env python3
"""
Moore-TDMPC2 代理训练的独立脚本
基于TDMPC2原始训练脚本修改，适配Moore架构
"""

import os
# 禁用 Kineto，避免所有 RecordFunction 的内部断言
os.environ["KINETO_ENABLED"] = "0"

import sys
import time
import warnings
import torch
from tqdm import tqdm

# Enable CUDA launch blocking to get better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 配置环境变量
os.environ['MUJOCO_GL'] = "osmesa"
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
warnings.filterwarnings('ignore')
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))

import torch
import hydra
from termcolor import colored
import numpy as np
from omegaconf import OmegaConf
from mujoco_py import MjSim, load_model_from_path
# 确保正确导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入TDMPC2基础组件
from TDMPC2.algorithms.common.parser import parse_cfg
from TDMPC2.algorithms.common.seed import set_seed
from TDMPC2.algorithms.common.buffer import Buffer
from TDMPC2.algorithms.envs import make_env 
from TDMPC2.algorithms.common.logger import Logger

# 导入Moore专用组件
from moore_tdmpc.world_model import MooreWorldModel
from moore_tdmpc.train_moore_agent import MooreTDMPC2, update_cfg_from_env

import torch._dynamo as dynamo
dynamo.reset()
from moore_tdmpc.train_moore_agent import MooreTDMPC2


# -------------------------------------------------
# 配置PyTorch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

# ────────────────────────────────────────────────────────────────
#  Inductor / torch.compile 预热工具
#  - 首次 run 时编译核心内核，后续 epoch / eval 复用缓存，省去数百秒开销
# ----------------------------------------------------------------
def _warm_up_inductor(world_model, cfg):
    """触发 compile。只调用一次即可。"""
    world_model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        B = 2  # 任意>1 即可，保证张量维度合法
        z = torch.zeros(B, cfg.latent_dim, device=cfg.device)
        a = torch.zeros(B, cfg.action_dim, device=cfg.device)

        # 触发四条核心路径的编译
        world_model.next(z, a)                     # dynamics
        world_model.Q(z, a, return_type='mean')    # Q
        world_model.reward(z, a)                   # reward
        world_model.pi(z)                          # policy
    torch.cuda.synchronize()
    world_model.train()

@hydra.main(config_name='moore_config', config_path='..')
def train(cfg):
    """
    Moore-TDMPC2训练入口点
    
    Args:
        cfg: hydra配置对象
    """
    print(colored("\n===== 初始化训练 =====", "yellow", attrs=["bold"]))
    
    # 验证基本条件
    assert torch.cuda.is_available(), "CUDA is required for training"
    assert cfg.steps > 0, '必须训练至少1步'
    
    # 解析配置并设置种子
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    
    print(colored('工作目录:', 'yellow', attrs=['bold']), cfg.work_dir)
    
    # 创建环境
    env = make_env(cfg)
    
    # 更新配置
    cfg = update_cfg_from_env(cfg, env)
    
    # 初始化组件
    from TDMPC2.algorithms.trainer.offline_trainer import OfflineTrainer
    from TDMPC2.algorithms.trainer.online_trainer import OnlineTrainer
    
    trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
    
    agent = MooreTDMPC2(cfg, env)

    # ============================================================
    # 在正式训练 / 评估开始前预编译 Inductor 内核（只跑一次）
    # ============================================================
    print("\033[93m[Warm-Up] compiling Inductor kernels …\033[0m")
    _warm_up_inductor(agent.model, cfg)
    print("\033[92m[Warm-Up] done — subsequent epochs will skip compile time.\033[0m")
    
    trainer = trainer_cls(
        cfg=cfg,
        env=env,
        agent=agent,
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    
    # 开始训练
    print(colored("\n开始Moore-TDMPC2训练...", "green", attrs=["bold"]))
    start_time = time.time()
    
    # 运行训练
    trainer.train()
    
    # 计算训练时间
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(colored(f"\n训练完成！总用时: {hours}小时 {minutes}分钟 {seconds}秒", "green", attrs=["bold"]))


if __name__ == '__main__':
    try:
        train() 
    except Exception as e:
        print(colored(f"训练过程中发生错误: {e}", "red"))
        raise 