import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm
from termcolor import colored

from common.buffer import Buffer
from trainer.base import Trainer

try:
	from utils.profiler import profiled, print_summary, get_profiling_status
	HAS_PROFILER = True
except ImportError:
	HAS_PROFILER = False
	# Create dummy decorator if profiler not available
	def profiled(func):
		return func


class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
	

	@contextmanager
	def _batch_first_sampler(self):
		orig_sample = self.buffer.sample

		def sample_batch_first(*args, **kwargs):
			obs, act, rew, *rest = orig_sample(*args, **kwargs)
			task = rest[0] if rest else None

			# 1) 如果 obs/act/rew 是 time-first，就转成 batch-first
			if (obs.dim()==3 and act.dim()==3 and rew.dim()==3
				and obs.size(0)==act.size(0)+1
				and act.size(0)==rew.size(0)):
				obs = obs.permute(1,0,2).contiguous()   # → [B,T+1,D]
				act = act.permute(1,0,2).contiguous()   # → [B,T,A]
				rew = rew.permute(1,0,2).contiguous()   # → [B,T,1]

			# 2) 统一 task → [B] （一维）
			if task is not None:
				B, T, _ = act.shape

				# 原始可能是 [T,B,1] / [T,B] / [B,T] / [B,1] / [B]
				if   task.dim()==3 and task.shape[:2]==(T,B):
					t_bT1 = task.permute(1,0,2)         # [B,T,1]
					tid   = t_bT1[:,0,0]                # [B]
				elif task.dim()==2 and task.shape==(T,B):
					tid   = task.permute(1,0)[:,0]      # [B]
				elif task.dim()==2 and task.shape==(B,T):
					tid   = task[:,0]                   # [B]
				elif task.dim()==2 and task.shape==(B,1):
					tid   = task[:,0]                   # [B]
				elif task.dim()==1 and task.shape==(B,):
					tid   = task                        # [B]
				else:
					raise ValueError(f"Unsupported task shape {tuple(task.shape)}")

				task = tid  # **一维 [B]**, 核心改动

			return (obs, act, rew, task) if task is not None else (obs, act, rew)

		self.buffer.sample = sample_batch_first
		try:
			yield
		finally:
			self.buffer.sample = orig_sample


	
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			ep_rewards, ep_successes = [], []
			for _ in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
				while not done:
					torch.compiler.cudagraph_mark_step_begin()
					action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					obs, reward, done, info = self.env.step(action)
					ep_reward += reward; t += 1
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
			results.update({
				f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
				f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),
			})
		return results
	
	def _load_dataset(self):
		"""Load dataset for offline training (原始 .pt 文件方式)."""
		fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
		fps = sorted(glob(str(fp)))
		assert len(fps) > 0, f'No data found at {fp}'
		print(f'Found {len(fps)} files in {fp}')
		if len(fps) < (20 if self.cfg.task == 'mt80' else 4):
			print(f'WARNING: expected 20 files for mt80, 4 for mt30, found {len(fps)}.')

		# 配置 buffer
		_cfg = deepcopy(self.cfg)
		_cfg.episode_length = 101 if self.cfg.task == 'mt80' else 501
		_cfg.buffer_size   = 550_450_000 if self.cfg.task == 'mt80' else 345_690_000
		_cfg.steps         = _cfg.buffer_size
		self.buffer = Buffer(_cfg)

		# 加载数据
		for pt_file in tqdm(fps, desc='Loading data'):
			td = torch.load(pt_file, weights_only=False)
			assert td.shape[1] == _cfg.episode_length, \
				f'Expected episode length {td.shape[1]} ≟ {_cfg.episode_length}'
			self.buffer.load(td)

		expected_eps = _cfg.buffer_size // _cfg.episode_length
		if self.buffer.num_eps != expected_eps:
			print(f'WARNING: buffer has {self.buffer.num_eps} eps, expected {expected_eps}.')

	@profiled
	def train(self):
		"""Train a TD-MPC2 agent."""
		if HAS_PROFILER:
			print(colored("OfflineTrainer: Starting training with profiling enabled", "green"))
			get_profiling_status()
			
		assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
			'Offline training only supports multitask training with mt30 or mt80 task sets.'
		# 1) 先加载数据集
		self._load_dataset()

		# 2) 关闭 TaskEncoder 的 checkpoint，以免在 encode 时引入维度错乱
		try:
			self.agent.model.task_encoder.use_checkpoint = False
			print("[OfflineTrainer] Disabled TaskEncoder gradient checkpointing")
		except Exception:
			pass

		# 3) 「打补丁」—— 把 batch‑first 转成 time‑first，encode 完再还原
		orig_encode = self.agent.model.encode

		def encode_batch_first(obs, task=None):
			# 只处理 3-D： [B, T+1, obs_dim]
			if obs.dim() == 3:
				B, Tp1, D_obs = obs.shape
				# → time-first
				obs_tf = obs.permute(1, 0, 2).contiguous()  # [T+1, B, D_obs]

				if task is not None:
					# 先确保 task 是 [B,1]
					if task.dim()==1 and task.shape[0]==B:           # [B]
						task_b1 = task.unsqueeze(1)                  # → [B,1]
					elif task.dim()==2 and task.shape==(B,1):        # [B,1]
						task_b1 = task
					else:
						raise ValueError(f"encode_batch_first: unsupported task shape {task.shape}")

					# → [T+1, B, 1]
					task_tf = task_b1.unsqueeze(0).expand(Tp1, B, 1)
					# 调用原始 time-first encode
					v_tf = orig_encode(obs_tf, task_tf)
				else:
					v_tf = orig_encode(obs_tf)

				# → back to batch-first: [B, T+1, latent_dim]
				return v_tf.permute(1, 0, 2).contiguous()

			# 2-D case 直接走原来
			return orig_encode(obs, task)

		self.agent.model.encode = encode_batch_first

		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		with self._batch_first_sampler():          # ← 仅 offline 生效
			for i in range(self.cfg.steps):
				
				if HAS_PROFILER and i % 1000 == 0:
					print(colored(f"OfflineTrainer: Training iteration {i}/{self.cfg.steps}", "cyan"))
					get_profiling_status()

				# Update agent
				train_metrics = self.agent.update(self.buffer)

				# Evaluate agent periodically
				if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
					metrics = {
						'iteration': i,
						'total_time': time() - self._start_time,
					}
					metrics.update(train_metrics)
					if i % self.cfg.eval_freq == 0:
						metrics.update(self.eval())
						self.logger.pprint_multitask(metrics, self.cfg)
						if i > 0:
							self.logger.save_agent(self.agent, identifier=f'{i}')
					self.logger.log(metrics, 'pretrain')
			
		self.logger.finish(self.agent)
		
		# Print profiling results if profiler is available
		if HAS_PROFILER:
			print(colored("\n===== OfflineTrainer Profiling Summary =====", "yellow", attrs=["bold"]))
			print_summary()
