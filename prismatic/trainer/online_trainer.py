from time import time

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tensordict.tensordict import TensorDict

from .base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()
        self._tds = []  # Initialize _tds to prevent AttributeError
        
        # Restore training state from agent if checkpoint was loaded
        if hasattr(self.agent, '_training_state'):
            state = self.agent._training_state
            checkpoint_step = state.get('step', 0)
            checkpoint_episode = state.get('episode', 0)
            
            # Check if force_step is specified and valid
            force_step = getattr(self.cfg, 'force_step', None)
            if force_step is not None and force_step != '???':
                # Use forced step count
                self._step = force_step
                self._ep_idx = checkpoint_episode  # Keep episode count from checkpoint
                print(f"🚀 FORCED step override: checkpoint={checkpoint_step} → forced={force_step}")
                print(f"   Episode count from checkpoint: {checkpoint_episode}")
            else:
                # Use checkpoint step count (original behavior)
                self._step = checkpoint_step
                self._ep_idx = checkpoint_episode
                print(f"📦 Restored from checkpoint: step={checkpoint_step}, episode={checkpoint_episode}")
            
            # Adjust start time to account for previous training time
            prev_time = state.get('start_time', 0)
            if prev_time > 0:
                self._start_time = time() - prev_time
                
            # Clean up the temporary training state
            delattr(self.agent, '_training_state')
        else:
            # Check if force_step is specified for fresh training
            force_step = getattr(self.cfg, 'force_step', None)
            if force_step is not None and force_step != '???':
                self._step = force_step
                print(f"🚀 FORCED step for fresh training: {force_step}")

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if self.cfg.save_video:
                # self.logger.video.save(self._step)
                self.logger.video.save(self._step, key='results/video')
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )

    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(1,),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, True
        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Save agent every 500k steps
            if self._step > 0 and self._step % 500000 == 0:
                # Set training state on agent before saving
                self.agent.set_training_state(
                    step=self._step,
                    episode=self._ep_idx, 
                    start_time=time() - self._start_time
                )
                self.logger.save_agent(self.agent, identifier=f"step_{self._step}")
                print(f"Saved agent checkpoint at step {self._step}")

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                # Only process episode metrics if we have trajectory data
                if self._step > 0 and len(self._tds) > 1:
                    train_metrics.update(
                        episode_reward=torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum(),
                        episode_success=info["success"],
                    )
                    train_metrics.update(self.common_metrics())

                    results_metrics = {'return': train_metrics['episode_reward'],
                                       'episode_length': len(self._tds[1:]),
                                       'success': train_metrics['episode_success'],
                                       'success_subtasks': info['success_subtasks'],
                                       'step': self._step,}
                
                    self.logger.log(train_metrics, "train")
                    self.logger.log(results_metrics, "results")
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self.env.reset()[0]
                self._tds = [self.to_td(obs)]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            self._tds.append(self.to_td(obs, action, reward))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                # Only update if buffer has data
                if self.buffer.num_eps > 0:
                    if self._step == self.cfg.seed_steps:
                        num_updates = self.cfg.seed_steps
                        print("Pretraining agent on seed data...")
                    else:
                        num_updates = 1
                    for _ in range(num_updates):
                        _train_metrics = self.agent.update(self.buffer)
                    train_metrics.update(_train_metrics)
                else:
                    print(f"Skipping agent update at step {self._step}: buffer empty (num_eps={self.buffer.num_eps})")

            self._step += 1

        # Set final training state before saving final checkpoint
        self.agent.set_training_state(
            step=self._step,
            episode=self._ep_idx,
            start_time=time() - self._start_time
        )
        self.logger.finish(self.agent) 