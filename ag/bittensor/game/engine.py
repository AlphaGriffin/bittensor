#!/usr/bin/python3
"""
Bittensor.
by: AlphaGriffin
"""

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2018, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.1"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Beta"

#////////////////// | Imports | \\\\\\\\\\\\\\\#
# generic
import os, sys, time, datetime, collections
from decimal import Decimal as D

# datasets
import pandas as pd

# Stats
from ag.bittensor.game.stats import Stats

class GameEngine(object):
    """
    The Game Engine is the mechinizm that controls buying and selling,
    paper or real.
    """

    def __init__(self, options):
        self.options = options
        if True:
            self.options.papertrader = True

        # Hold the Game Stats Here.
        self.stats = Stats(options)

    def setup(self, dataframe):
        # build up the starting stats for the coin.
        # figure for start btc from starting usd
        # figure total coin value from btc value

        class New_StockEnv(object):
            def __init__(self, dataframe):
                self.data = dataframe
                self.step = 0

            def reset(self):
                data = self.data[self.step]
                return data

            def take_step(self):
                self.step += 1
                if self.step >= 239:
                    done = True
                else:
                    done = False
                return self.data[self.step-1], self.data[self.step], done

        self.data = New_StockEnv(dataframe)
        self.action_space = ['b', 's', 'n']
        self.n_actions = len(self.action_space)
        self.action_history = []
        self.price_history = pd.Series(len(self.data))

        return self.data

    def take_action(self, action):
        last_data, next_data, done = self.data.take_step()
        last_price = last_data[0]
        this_price = next_data[0]

        # update global frame for final graph
        self.action_history.append(action)
        self.price_history.append(last_price)

        last_action = self.action_history[-1]
        # Reward ZONE!
        # .5 for a win + percentage of gain
        # -.5 for a loss + percentage of loss
        # .15 for do nothing
        # 0 for buy
        # 0 for sell
        # .05 for positive balance
        # -.05 for negitive balance
        reward = 0

        # Actionable Options
        if action == 0:  # BUY!
            if len(self.stats.buy_entries) < 3:  # Dont have more than 3 open buys!
                self.stats.buy_entries.append(this_price)

        elif action == 1:  # SELL!
            if self.stats.is_position_open:
                lowest_buy_price = min(self.stats.buy_entries)
                if D(this_price) > D(self.stats.cur_position) * 1.025:  # This is a winning Trade... Rewards!
                    asdf = 0


        else:
            pass   # DO NOTHING

        # Calculate Reward


        return next_data, reward, done


    def create_graph(self):
        return True

    def main(self):
        """Sanity Check."""
        if self.options:
            return True
        return False
