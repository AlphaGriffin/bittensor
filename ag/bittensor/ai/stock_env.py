#!/usr/bin/python3
"""
Stocks Environment.
by: HubFire
Updated by: Ruckusist
"""

import pandas as pd

__author__ = "@HubFire"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["HubFire", "Eric Petersen", "Shawn Wilson"]
__license__ = "***"
__version__ = "0.0.1"
__maintainer__ = "https://github.com/HubFire"
__email__ = "https://github.com/HubFire"
__status__ = "Beta"

class StockEnv(object):
	def __init__(self, data):
		self.action_space = ['b','s','n']
		self.n_actions = len(self.action_space)
		self.data = data
		self.step = 0
		self.hold = 0
		self.cash = 1
		self.last_buy_price = 0
		self.has_bought = False
		self.winning_trades = 0
		self.losing_trades = 0

	def reset(self):
		data = self.data[self.step]
		self.hold = 0
		self.cash = 0
		return data

	def gostep(self,action):
		cash = self.cash
		hold = self.hold
		self.step +=1
		s = self.data[self.step-1]
		current_price =s[0]
		reward = 0

		if action == 0: # buy
			if not self.has_bought:
				self.hold += 1
				self.cash -= current_price
				self.last_buy_price = current_price
				self.has_bought = True
		elif action == 1: #sell
			if self.has_bought:
				self.has_bought = False
				self.last_buy_price = 0
				self.hold -= 1
				self.cash += current_price
				if current_price > self.last_buy_price:
					self.winning_trades += 1
					reward = 1
				else:
					self.losing_trades += 1
					reward = -1
		else:
			reward = .1
			# pass  # nothing

		if self.step ==239:
			done = True
		else:
			done = False
		s_ = self.data[self.step]
		"""
		new_price = s_[0]
		new_pro = s_[0] * self.hold + self.cash
		old_pro = current_price * hold+cash
		reward = new_pro - old_pro
		"""
		return s_, reward, done

# df = pd.read_csv('./data.csv')
# env = StockEnv(df)
# s0= env.reset()
# s1,reward,done = env.gostep(1)
# s2,reward,done = env.gostep(1)
# print reward
# #print done
