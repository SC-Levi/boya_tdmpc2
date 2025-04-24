import torch
import torch.nn.functional as F
import numpy as np

from TDMPC2.algorithms.common import math
from TDMPC2.algorithms.common.scale import RunningScale
from TDMPC2.algorithms.common.moore_world_model import MooreWorldModel
from TDMPC2.algorithms.common.layers import api_model_conversion
from tensordict import TensorDict
from TDMPC2.algorithms import TDMPC2


class MooreTDMPC2(TDMPC2):
    """
    集成Moore混合专家架构的TD-MPC2 agent
    扩展原始的TDMPC2类，使用MooreWorldModel替代标准WorldModel
    
    Args:
        cfg: 配置对象，除标准TDMPC2配置外，还支持以下Moore特有配置：
            - n_experts: 专家数量，默认4
            - temperature: 专家权重的softmax温度，默认1.0
            - use_softmax: 是否使用softmax归一化专家权重，默认True
            - expert_hidden_dims: 专家网络隐藏层维度
            - debug_task_emb: 是否启用任务嵌入调试输出
    """

    def __init__(self, cfg):
        # 不调用父类__init__，而是复制其代码并替换WorldModel为MooreWorldModel
        super(TDMPC2, self).__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0')
        
        # 使用MooreWorldModel代替标准WorldModel
        self.model = MooreWorldModel(cfg).to(self.device)
        
        # 设置优化器，确保Moore任务嵌入模块参数正确更新
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._moore_task_emb.parameters() if self.cfg.multitask else []}
        ], lr=self.cfg.lr, capturable=True)
        
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)
        self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
        
        if getattr(cfg, 'compile', False):
            print('Compiling update function with torch.compile...')
            self._update = torch.compile(self._update, mode="reduce-overhead")
    
    def get_expert_stats(self):
        """
        获取任务-专家分配的统计信息
        
        Returns:
            专家使用统计信息，包括平均权重和任务计数
        """
        if hasattr(self.model, 'get_expert_stats'):
            return self.model.get_expert_stats()
        return None
    
    def save(self, fp):
        """
        保存模型状态，包括Moore专家网络状态
        
        Args:
            fp: 文件路径
        """
        expert_stats = self.get_expert_stats()
        state_dict = {
            "model": self.model.state_dict(),
            "moore_stats": expert_stats
        }
        torch.save(state_dict, fp)
    
    def update(self, buffer):
        """
        训练更新函数，添加Moore专家使用统计信息
        
        Args:
            buffer: 回放缓冲区
            
        Returns:
            训练指标
        """
        metrics = super().update(buffer)
        
        # 添加Moore专家网络的统计信息
        if self.cfg.multitask:
            expert_stats = self.get_expert_stats()
            if expert_stats and expert_stats["avg_weights"] is not None:
                # 计算任务-专家权重的平均熵
                weights = expert_stats["avg_weights"]
                entropy = -np.sum(weights * np.log(weights + 1e-10), axis=1).mean()
                metrics["expert_entropy"] = entropy
                
                # 计算专家权重方差，指示任务间专家分配差异
                weight_var = np.var(weights, axis=0).mean()
                metrics["expert_variance"] = weight_var
        
        return metrics
            
    # 其余方法继承自TDMPC2基类 