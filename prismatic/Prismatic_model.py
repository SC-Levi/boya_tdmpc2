import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.memory_monitor import MemoryMonitor
from common.runtime_memory_manager import RuntimeMemoryManager, create_memory_manager
from tensordict import TensorDict
from common import layers


class PrismaticModel(torch.nn.Module):
	"""
	Prismatic agent (formerly Moore-TDMPC). Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		
		# 初始化内存监控器
		self.memory_monitor = MemoryMonitor(log_interval=getattr(cfg, 'monitor_mem_interval', 1000))
		
		# 初始化运行时内存管理器
		self.runtime_memory_manager = None
		if cfg.use_moe:
			self.runtime_memory_manager = create_memory_manager(
				agent=self, 
				auto_start=getattr(cfg, 'enable_runtime_memory_manager', True)
			)
			print("✅ 运行时内存管理器已启动")
			print(f"   - 进程ID: {self.runtime_memory_manager.log_file.split('_')[-1].split('.')[0] if self.runtime_memory_manager.log_file else 'N/A'}")
			print("   - 使用信号控制: kill -USR1 <pid> (清理), kill -USR2 <pid> (状态)")
			print("   - 控制文件: /tmp/prismatic_memory_control")
		
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self.register_buffer('_prev_mean', torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		save_dict = {"model": self.model.state_dict()}
		
		# Add training state if it exists
		if hasattr(self, '_training_state'):
			save_dict["training_state"] = self._training_state
			
		torch.save(save_dict, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
				fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])
		
		# Load training state if it exists
		if "training_state" in state_dict:
			self._training_state = state_dict["training_state"]
			print(f"📦 Found training state in checkpoint: step={self._training_state.get('step', 0)}, episode={self._training_state.get('episode', 0)}")

	def set_training_state(self, step, episode, start_time):
		"""
		Set training state to be saved with the model.
		
		Args:
			step (int): Current training step
			episode (int): Current episode count  
			start_time (float): Total training time elapsed
		"""
		self._training_state = {
			'step': step,
			'episode': episode,
			'start_time': start_time
		}

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None, goal=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).
			goal (torch.Tensor): Goal latent state for hierarchical control.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task, goal=goal).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task, goal=None):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			if goal is not None:
				# Add goal-directed reward: negative distance to goal
				# Ensure goal has correct shape [num_samples, latent_dim]
				if goal.dim() == 2 and goal.shape[0] == 1:
					# goal is [1, latent_dim], expand to [num_samples, latent_dim]
					expanded_goal = goal.expand(z.shape[0], -1)
				else:
					# goal already has correct shape
					expanded_goal = goal
				goal_reward = -torch.norm(z - expanded_goal, dim=-1, keepdim=True)
				env_reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
				reward = env_reward + 0.1 * goal_reward  # Combine environment and goal rewards
			else:
				reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None, goal=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			obs (torch.Tensor): Observation to plan from.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).
			goal (torch.Tensor): Goal latent state for hierarchical control.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions (with goal conditioning)
			value = self._estimate_value(z, actions, task, goal).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _update(self, obs, action, reward, terminated, task=None):
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task)

		# Prepare for update
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# 处理MoE辅助损失
		if self.cfg.use_moe:
			moe_aux_loss = self.model.get_moe_aux_loss()
			total_loss = total_loss + 0.01 * moe_aux_loss  # 添加辅助损失
			# 重置MoE辅助损失累积器
			self.model.zero_moe_aux_loss()

		# Update policy
		pi_info = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, terminated, task = buffer.sample()
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		
		# 执行实际更新
		info = self._update(obs, action, reward, terminated, **kwargs)
		
		# 每次更新后都清除门控历史，防止内存泄漏
		self.clear_gate_history()
		
		# 定期强制垃圾回收（可选，用于严重内存泄漏情况）
		if hasattr(self, '_update_count'):
			self._update_count += 1
		else:
			self._update_count = 1
			
		# 内存监控和清理
		memory_stats = self.memory_monitor.log_memory(step=self._update_count)
		if memory_stats:
			print(self.memory_monitor.format_memory_stats(memory_stats))
			
		if self._update_count % 100 == 0:  # 每100次更新强制清理
			self.memory_monitor.cleanup_memory()
			
			# 检查内存泄漏
			if self.memory_monitor.check_memory_leak(threshold_increase_gb=2.0):
				print(f"⚠️  Potential memory leak detected at step {self._update_count}")
				print("Peak memory usage:", self.memory_monitor.get_peak_memory())
				# 进行更激进的清理
				self.memory_monitor.cleanup_memory(aggressive=True)
		
		return info
		
	def plot_expert_gating(self, moe_block, save_path=None):
		"""
		Plot expert gating weights and entropy over time.
		
		Args:
			moe_block: MoEBlock instance with recorded gate_history
			save_path: Optional path to save the plot
		"""
		if not moe_block.gate_history:
			print("No gating history recorded")
			return
			
		# Stack into a [T, K] tensor
		hist = torch.cat(moe_block.gate_history, dim=0).numpy()
		T, K = hist.shape
		
		# Import matplotlib for plotting
		import matplotlib.pyplot as plt
		import numpy as np
		
		plt.figure(figsize=(10, 6))
		
		# Plot heatmap
		plt.subplot(2, 1, 1)
		plt.imshow(hist.T, aspect='auto', cmap='viridis')
		plt.ylabel('Expert index')
		plt.title('Gate weights')
		plt.colorbar()
		
		# Plot entropy
		entropy = -(hist * np.log(hist + 1e-8)).sum(axis=1)  # [T]
		plt.subplot(2, 1, 2)
		plt.plot(entropy, '-o', color='C1', linewidth=2)  # Use different color and thicker line
		plt.xlabel('Time step')
		plt.ylabel('Entropy')
		plt.title('Gating entropy over time')
		plt.ylim(0, np.log(K) * 1.05)  # Add 5% padding above log(K) for better visibility
		
		plt.tight_layout()
		
		if save_path:
			plt.savefig(save_path)
		else:
			plt.show()
		
		return hist, entropy
		
	def clear_gate_history(self):
		"""Clear gate history from all MoEBlock instances in the model (safe: do not clear _last_entropy)"""
		if hasattr(self.model, '_dynamics') and hasattr(self.model._dynamics, 'gate_history'):
			self.model._dynamics.gate_history.clear()
		if hasattr(self.model, '_reward') and hasattr(self.model._reward, 'gate_history'):
			self.model._reward.gate_history.clear()
		# Do NOT clear _last_entropy for safety
