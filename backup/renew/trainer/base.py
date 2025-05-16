class Trainer:
	"""Base trainer class for moore-TD-MPC."""

	def __init__(self, cfg, env, agent, buffer, logger):
		self.cfg = cfg
		print(self.cfg)
		self.env = env
		self.agent = agent
		self.buffer = buffer
		self.logger = logger
		print('Architecture:', self.agent.model)

	def eval(self):
		"""Evaluate a moore-TD-MPC agent."""
		raise NotImplementedError

	def train(self):
		"""Train a moore-TD-MPC agent."""
		raise NotImplementedError
