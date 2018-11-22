from __future__ import print_function, division
from builtins import range

import numpy as np


class Grid3D: # Environment
	def __init__(self, shape, start):
		if len(shape) != 3:
			raise ValueError("Incorrect value for Shape. Shape must be a tuple of 3 dimensions.")
		if len(start) != 3:
			raise ValueError("Incorrect value for Start. Start must be a list of  3 dimensions.")
		self.shape = shape
		self.pos = start
		self.action_space = [0, 1, 2, 3, 4, 5]

	def set(self, rewards, actions, obey_prob):
		# rewards should be a dict of: (i, j, k): r (row, col, dep): reward
		# actions should be a dict of: (i, j, k): A (row, col, dep): list of possible actions
		# action space is [0, 1, 2, 3, 4, 5]
		self.rewards = rewards
		self.actions = actions
		self.obey_prob = obey_prob
	
	def action_space(self):
		return self.action_space

	def non_terminal_states(self):
		return self.actions.keys()

	def set_state(self, s):
		if len(s) != 3:
			raise ValueError("Incorrect value for state. State must be a tuple of 3 dimensions.")
		self.pos = list(s)

	def current_state(self):
		return tuple(self.pos)

	def is_terminal(self, s):
		return s not in self.actions

	def stochastic_move(self, action):
		p = np.random.random()
		if p <= self.obey_prob:
			return action
		if action == 0 or action == 1:
			return np.random.choice([2, 3, 4, 5])
		elif action == 2 or action == 3:
			return np.random.choice([0, 1, 4, 5])
		elif action == 4 or action == 5:
			return np.random.choice([0, 1, 2, 3])

	def move(self, action):
		actual_action = self.stochastic_move(action)
		if actual_action in self.actions[tuple(self.pos)]:
			if actual_action == 0:
				self.pos[0] -= 1
			elif actual_action == 1:
				self.pos[0] += 1
			elif actual_action == 2:
				self.pos[1] -= 1
			elif actual_action == 3:
				self.pos[1] += 1
			elif actual_action == 4:
				self.pos[2] -= 1
			elif actual_action == 5:
				self.pos[2] += 1
		return self.rewards.get(tuple(self.pos), 0)

	def check_move(self, action):
		pos = self.pos.copy()
		# check if legal move first
		if action in self.actions[tuple(self.pos)]:
			if action == 0:
				pos[0] -= 1
			elif action == 1:
				pos[0] += 1
			elif action == 2:
				pos[1] -= 1
			elif action == 3:
				pos[1] += 1
			elif action == 4:
				pos[2] -= 1
			elif action == 5:
				pos[2] += 1
		# return a reward (if any)
		reward = self.rewards.get(tuple(pos), 0)
		return (pos, reward)

	def get_transition_probs(self, action):
		# returns a list of (probability, reward, s') transition tuples
		probs = []
		state, reward = self.check_move(action)
		probs.append((self.obey_prob, reward, tuple(state)))
		disobey_prob = 1 - self.obey_prob
		if not (disobey_prob > 0.0):
			return probs
		if action == 0 or action == 1:
			state, reward = self.check_move(2)
			probs.append((disobey_prob / 2, reward, tuple(state)))
			state, reward = self.check_move(3)
			probs.append((disobey_prob / 2, reward, tuple(state)))
			state, reward = self.check_move(4)
			probs.append((disobey_prob / 2, reward, tuple(state)))
			state, reward = self.check_move(5)
			probs.append((disobey_prob / 2, reward, tuple(state)))
		elif action == 2 or action == 3:
			state, reward = self.check_move(0)
			probs.append((disobey_prob / 2, reward, tuple(state)))
			state, reward = self.check_move(1)
			probs.append((disobey_prob / 2, reward, tuple(state)))
			state, reward = self.check_move(4)
			probs.append((disobey_prob / 2, reward, tuple(state)))
			state, reward = self.check_move(5)
			probs.append((disobey_prob / 2, reward, tuple(state)))
		elif action == 4 or action == 5:
			state, reward = self.check_move(0)
			probs.append((disobey_prob / 2, reward, tuple(state)))
			state, reward = self.check_move(1)
			probs.append((disobey_prob / 2, reward, tuple(state)))
			state, reward = self.check_move(2)
			probs.append((disobey_prob / 2, reward, tuple(state)))
			state, reward = self.check_move(3)
			probs.append((disobey_prob / 2, reward, tuple(state)))
		return probs

	def game_over(self):
		# returns true if game is over, else false
		# true if we are in a state where no actions are possible
		return tuple(self.pos) not in self.actions

	def all_states(self):
		# possibly buggy but simple way to get all states
		# either a position that has possible next actions
		# or a position that yields a reward
		return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid3D(obey_prob=1.0, step_cost=None):
	# define a grid that describes the reward for arriving at each state
	# and possible actions at each state
	# the grid looks like this
	# x means you can't go there
	# s means start position
	# number means reward at that state
	# .  .  .  1
	# .  x  . -1
	# s  .  .  .
	# obey_brob (float): the probability of obeying the command
	# step_cost (float): a penalty applied each step to minimize the number of moves (-0.1)
	g = Grid3D((3, 3, 3), [2, 0, 0])
	rewards = {(0, 0, 0): -1, (1, 1, 1): -1, (2, 2, 2): -1, (0, 2, 2): 1}
	actions = {
		(0, 0, 1): (1, 3, 4, 5),
		(0, 0, 2): (3, 4),
		(0, 1, 0): (1, 2, 3, 5),
		(0, 1, 1): (1, 2, 3, 4, 5),
		(0, 1, 2): (1, 2, 3, 4),
		(0, 2, 0): (2, 5),
		(0, 2, 1): (1, 2, 4, 5),
		(1, 0, 0): (0, 1, 3, 5),
		(1, 0, 1): (0, 1, 3, 4, 5),
		(1, 0, 2): (0, 1, 3, 4),
		(1, 1, 0): (0, 1, 2, 5),
		(1, 1, 2): (0, 1, 3, 4),
		(1, 2, 1): (0, 1, 2, 5),
		(1, 2, 2): (0, 1, 2, 4),
		(2, 0, 0): (0, 3, 5),
		(2, 0, 1): (0, 3, 4, 5),
		(2, 0, 2): (3, 4),
		(2, 1, 0): (0, 2, 3, 5),
		(2, 1, 1): (0, 2, 3, 4, 5),
		(2, 1, 2): (0, 2, 3, 4),
		(2, 2, 0): (2, 5),
		(2, 2, 1): (0, 2, 4, 5)
	}
	g.set(rewards, actions, obey_prob)
	if step_cost is not None:
		for i in range(3):
			for j in range(3):
				for k in range(3):
					if (i, j, k) not in g.rewards.keys():
						g.rewards.update({(i, j, k): step_cost})
	return g

