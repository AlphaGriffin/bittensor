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

# Math
import numpy as np
from decimal import Decimal as D


class Stats(object):
    """
    A mechinizm that holds the games variables, positions, data,
    can parse the options for limitations, holds all golbal booleans.
    """

    def __init__(self, options):
        """Build out all the stats here."""
        # build Objects
        self.__options = options

        if options.papertrader:
            self.reset_paperTrader()


    def reset_paperTrader(self):
        # Actual Stats.
        self.starting_usd = 1000
        self.starting_btc = 0
        self.starting_coin = 0

        self.current_usd = 0
        self.current_btc = 0
        self.current_coin = 0

        # self.current_percentage = D(self.current_btc / self.starting_btc) - 1

        self.buy_entries = []
        self.sell_entries = []

        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = self.losing_trades + self.winning_trades

    @property
    def current_percentage(self):
        return D(self.current_btc / self.starting_btc) - 1

    @property
    def cur_position(self):
        return np.mean(self.buy_entries)

    @property
    def is_position_open(self):
        return bool(self.buy_entries)
