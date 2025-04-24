import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch
import sys

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(project_root))  # 添加上级目录以导入moore_tdmpc

import hydra
from termcolor import colored

from algorithms.common.parser import parse_cfg
from algorithms.common.seed import set_seed
from algorithms.common.buffer import Buffer
from algorithms.envs import make_env
from algorithms.trainer.offline_trainer import OfflineTrainer
from algorithms.trainer.online_trainer import OnlineTrainer
from algorithms.common.logger import Logger

# 从moore_tdmpc目录中导入MooreTDMPC2
from moore_tdmpc.tdmpc2 import MooreTDMPC2

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
    """
    训练Moore-TDMPC2多任务混合专家代理的脚本。
    
    参数:
        `task`: 任务名称（如mt30/mt80表示多任务训练）
        `model_size`: 模型大小，必须是 `[1, 5, 19, 48, 317]` 之一 (默认: 5)
        `steps`: 训练步数（默认: 10M）
        `seed`: 随机种子 (默认: 1)
        
    Moore专有参数：
        `n_experts`: 专家数量 (默认: 4)
        `temperature`: 专家权重softmax温度参数 (默认: 1.0)
        `use_softmax`: 是否使用softmax归一化专家权重 (默认: True)
        
    使用示例:
    ```
        $ python train_moore.py task=mt80 model_size=48 n_experts=8
        $ python train_moore.py task=mt30 model_size=317 temperature=0.5
    ```
    """
    assert torch.cuda.is_available()
    assert cfg.steps > 0, '必须训练至少1步。'
    
    # 添加Moore架构默认配置
    cfg.n_experts = getattr(cfg, 'n_experts', 4)
    cfg.temperature = getattr(cfg, 'temperature', 1.0)
    cfg.use_softmax = getattr(cfg, 'use_softmax', True)
    cfg.debug_task_emb = getattr(cfg, 'debug_task_emb', False)
    
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored('工作目录:', 'yellow', attrs=['bold']), cfg.work_dir)
    print(colored('Moore配置:', 'green', attrs=['bold']), 
          f"专家数量={cfg.n_experts}, 温度={cfg.temperature}, Softmax={cfg.use_softmax}")

    trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
    trainer = trainer_cls(
        cfg=cfg,
        env=make_env(cfg),
        agent=MooreTDMPC2(cfg),  # 使用MooreTDMPC2替代TDMPC2
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    trainer.train()
    print('\n训练成功完成')


if __name__ == '__main__':
    train() 