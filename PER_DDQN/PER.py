import random
import numpy as np
from collections import namedtuple

from sumTree import SumTree
from config import Config

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Memory:  # stored as < s, a, s', r > in SumTree
	

	def __init__(self, capacity):
		self.tree = SumTree(capacity)
		self.capacity = capacity

		self.a = Config.A #hyperparameter used to reintroduce randomnes in the experience selection
		self.e = Config.E 
		self.beta = Config.BETA
		self.beta_increment_per_sampling = Config.BETA_INCREMENT

	def _get_priority(self, error):
		return (error + self.e) ** self.a

	def add(self, error, *args):
		p = self._get_priority(error)
		self.tree.add(p, Transition(*args))

	def sample(self, n):
		batch = []
		idxs = []
		segment = self.tree.total() / n
		priorities = []

		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

		for i in range(n):
			a = segment * i
			b = segment * (i + 1)

			s = random.uniform(a, b)
			(idx, p, data) = self.tree.get(s)
			priorities.append(p)
			batch.append(data)
			idxs.append(idx)

		sampling_probabilities = priorities / self.tree.total()
		is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
		is_weight /= is_weight.max()

		return batch, idxs, is_weight

	def update(self, idx, error):
		p = self._get_priority(error)
		self.tree.update(idx, p)