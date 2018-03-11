#!/usr/bin/python3
"""
Bittensor.
by: AlphaGriffin
"""

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2018, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.2"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Beta"

#////////////////// | Imports | \\\\\\\\\\\\\\\#
# generic
import os, sys, time, datetime, collections

# Math
import numpy as np
from decimal import Decimal as D

#Data Structure
import pandas as pd


class Stats(object):
    """
    A mechinizm that holds the games variables, positions, data,
    can parse the options for limitations, holds all global booleans.
    """

    def __init__(self, options=None):
        """Build out all the stats here."""
        # build Objects
        self.__options = options
        self._tradehistory = None
        self.reset_paperTrader()

    def reset_paperTrader(self):
        # Actual Stats.
        # RTBV = running total bitcoin value
        history_cols = [
            'Start_Time', "End_Time","Fees", "Open_Price",
            "Close_Price","Position","Amount_BTC",
            "Amount_Coin","Return_BTC","RTBV" 
        ]
        tradeHistory = pd.DataFrame(columns=history_cols)

        tradeHistory = tradeHistory.append({
            'Start_Time': datetime.datetime.now(),
            'End_Time': datetime.datetime.now(),
            'Fees': 0,
            'Open_Price': 0,
            'Close_Price': 0,
            'Position': 'Start',
            'Amount_BTC': 0,
            'Amount_Coin': 0,
            'Return_BTC': 0,
            'RTBV': 0.001,
        }, ignore_index=True)

        # tradeHistory.set_index('Start_time', inplace=True)
        self._tradehistory = tradeHistory

    @property
    def paperTrader(self):
        # self._tradehistory.set_index('Start_Time', inplace=True)
        return self._tradehistory

    @paperTrader.setter
    def paperTrader(self, value):
        self._tradehistory = self._tradehistory.append(value, ignore_index=True)

    def process_trades(self, dataframe):
        # DONE :: ADD IN FEES...
        # TODO :: TAKE 50% of all profits for taxes to a seperate account
        #      :: use the tax account for the lending bot.
        hist = self.paperTrader
        __buy__ = True
        trade = {
            "Start_Time": None,
            "End_Time": None,
            "Open_Price": 0,
            "Fees": 0,
            "Close_Price": None,
            "Position": 'None',
            "Signal": 'From_Mars',
            "Amount_BTC": None,  # hist['RTBV'][:-1].values * .1  # buy 10% stack
            "Amount_Coin": None,  # amount_btc * open_price
            "Return_BTC": None,  # amount_coin * close_price
            "RTBV": None,  # hist['RTBV'][:-1].values + return_btc
        }
        for signal in dataframe:
            if 'signal' not in signal:
                continue
            for i, e in enumerate(dataframe[signal]):
                if __buy__:
                    if e >= 1:
                        t = trade
                        t["Start_Time"] = dataframe.index[i]
                        t["Open_Price"] = dataframe['Close'].iloc[i]
                        t["Position"] = 'Long'
                        t["Signal"] = signal
                        t["Amount_BTC"] = hist['RTBV'].iloc[-1] * .1  # buy 10% stack
                        t["Fees"] = t["Amount_BTC"] * 0.025  # .25% all trades fee
                        t["Amount_Coin"] = (t["Amount_BTC"]-t["Fees"]) / t['Open_Price']
                        t['RTBV'] = hist['RTBV'].iloc[-1] - t["Amount_BTC"]

                        __buy__ = False
                else:
                    if e <= -1:
                        t["End_Time"] = dataframe.index[i]
                        t["Close_Price"] = dataframe['Close'].iloc[i]
                        t["Return_BTC"] = t["Amount_Coin"] * t['Close_Price']
                        second_fee = t["Return_BTC"] * 0.025
                        t["Return_BTC"] = t["Return_BTC"] - second_fee
                        t["Fees"] = t["Fees"] + second_fee
                        t['RTBV'] = t['RTBV'] + t["Return_BTC"]
                        __buy__ = True
                        self.paperTrader = t
                        del(t)
                if i >= len(dataframe)-1:
                    # print("finished")
                    try:
                        if t:
                            # print('Closing trade early..')
                            t["End_Time"] = dataframe.index[i]
                            t["Close_Price"] = dataframe['Close'].iloc[i]
                            t["Return_BTC"] = t["Amount_Coin"] * t['Close_Price']
                            second_fee = t["Return_BTC"] * 0.0025
                            t["Return_BTC"] = t["Return_BTC"] - second_fee
                            t["Fees"] = t["Fees"] + second_fee
                            t['RTBV'] = t['RTBV'] + t["Return_BTC"]
                            self.paperTrader = t
                            del(t)
                    except:
                        pass
        return self.trader_profits          

    @property
    def trader_profits(self):
        return (self._tradehistory['RTBV'].iloc[-1] / self._tradehistory['RTBV'].iloc[0]) - 1

    @property
    def all_fees_paid(self):
        return self._tradehistory['Fees'].sum()