import numpy as np
import gymnasium as gym
from envs.wrappers.timeout import Timeout


MYOSUITE_TASKS = {
	'myo-reach': 'myoHandReachFixed-v0',
	'myo-reach-hard': 'myoHandReachRandom-v0',
	'myo-pose': 'myoHandPoseFixed-v0',
	'myo-pose-hard': 'myoHandPoseRandom-v0',
	'myo-obj-hold': 'myoHandObjHoldFixed-v0',
	'myo-obj-hold-hard': 'myoHandObjHoldRandom-v0',
	'myo-key-turn': 'myoHandKeyTurnFixed-v0',
	'myo-key-turn-hard': 'myoHandKeyTurnRandom-v0',
	'myo-pen-twirl': 'myoHandPenTwirlFixed-v0',
	'myo-pen-twirl-hard': 'myoHandPenTwirlRandom-v0',
}


class MyoSuiteWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.camera_id = 'hand_side_inter'

	def reset(self):
		return self.env.reset()[0]

	def step(self, action):
		obs, reward, _, _, info = self.env.step(action.copy())
		info['success'] = info['solved']
		return obs, reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.sim.renderer.render_offscreen(
			width=384, height=384, camera_id=self.camera_id
		).copy()


def make_env(cfg):
	"""
	Make Myosuite environment.
	"""
	if not cfg.task in MYOSUITE_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	import myosuite
	from myosuite.utils import gym as gym_utils
	env = gym_utils.make(MYOSUITE_TASKS[cfg.task])
	env = MyoSuiteWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=100)
	#env.max_episode_steps = env._max_episode_steps
	return env
