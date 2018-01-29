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
import pandas as pd
import numpy as np

import ccxt
# import ag.bittensor.api.poloniex as Polo

class DataStruct(pd.DataFrame):
    """
    Added some functions to DataFrame.
    (1) fix milliseconds
    (2) undo fix
    (3) create a tailored superset for tensorflow input
    Source:
        http://blog.snapdragon.cc/2015/05/05/subclass-pandas-dataframe-to-save-custom-attributes/
    """

    def __init__(self, *args, **kw):
        super(DataStruct, self).__init__(*args, **kw)
        self.exchange = None
        self.pair = None
        self.start_date = None
        self.end_date = None

    @property
    def _constructor(self):
        return DataStruct

    @property
    def _constructor_sliced(self):
        return pd.Series

    def fix_time(self):
        df = self.copy()
        # check to see if 'time' exists and len of the df
        try:
            sLen = len(df['time'])
        except:
            print("Cant Fix the time!")
            return df
        # df['time_str'] = pd.Series('' * sLen)
        df['time_str'] = pd.Series(''*sLen)
        try:
            for i, e in enumerate(self['time']):

                if i % 1000 == 0:
                    print('processed {} of {}'.format(i, len(df['time'])))

                if e == 0:
                    df['time_str'][i] = 'NaN'
                    continue
                realtime = int(int(e) / 1000)
                time_str = datetime.datetime.fromtimestamp(realtime).strftime('|%Y-%m-%d %H:%M:%S|')
                df['time_str'][i] = time_str

        except:
            print('Cant Change datetime!')
            return df
        print('Changed to datetime format!')
        df.exchange = self.exchange
        df.pair = self.pair
        return df

    def remove_time(self):
        df = self
        if 'time_str' in df.columns:
            df = df.drop(['time_str'], axis=1)
            return df
        else:
            print("Cant remove time_str")
            return df

    def superset(self):
        """
        :return: A numpy array of the dataframe.
        """
        df = self
        cols = []
        for i in range(20):
            cols.append('close_{}'.format(i))
            cols.append('vol_{}'.format(i))

        # working_df = pd.DataFrame(columns=cols)
        working_df = DataStruct(columns=cols)
        for index in range(len(df)):
            # start once we have 20 samples!
            if index <= len(df) - 20:

                # we are using 20 data samples per line
                dataline = []
                for i in range(20):
                    time = df.loc[i + index]['time']
                    close = df.loc[i + index]['close']
                    vol = df.loc[i + index]['vol']
                    # print('{} | {} | close: {}, vol: {}'.format(i, exchange.iso8601(time), close, vol))
                    dataline.append(float(close))
                    dataline.append(float(vol))
                new_df = pd.DataFrame([dataline], columns=cols)
                working_df = working_df.append(new_df)
        return working_df  # .as_matrix()

    def mircoset(self):
        """
        :return: A numpy array of the dataframe.
        """
        df = self
        df = df[-20:]
        series = []
        for x, y in zip(df['close'], df['vol']):
            series.append(x)
            series.append(y)
        return series


    def myLen(self):
        df = self
        return len(df)

    @staticmethod
    def upgrade(dataframe):
        cols = [x for x in dataframe.columns]
        df = DataStruct(columns=cols)
        df.append(dataframe)
        return df


class DataHandler(object):
    """This Should handle the webside data collections; and other stuff too."""

    def __init__(self, options):
        self.data = DataStruct([[1,2,3,4,5], ['a','b','c','d','e']], columns=['time','that','other','thing1','thing2'])
        self.source = ccxt
        self.options = options

    def main(self):
        # new_dataframe =  self.data.fix_time()
        # print(new_dataframe)
        # newer_dataframe = self.data.remove_time()
        # print(newer_dataframe)
        # return new_dataframe
        return True

    def get_candles(self, exchange='poloniex', pair='BTC/USDT'):
        """
        :param exchange: this should be known in advance with api information in the config.
        :param pair: A known trading pair, try seaching for this first and insert programatically.
        :return: A custom pandas dataframe with a fix_time and superset functions added.
        """
        # add setup info for ccxt
        config = {'rateLimit': 3000,
                  'enableRateLimit': True,
                  # 'verbose': True,
                  }
        # this is a setup with no login.
        this_exchange = eval('ccxt.{}({})'.format(exchange, config))
        time.sleep(this_exchange.rateLimit / 1000)

        # this is the webcall for candlestick data
        OHLCVS = this_exchange.fetch_ohlcv(
            pair, '5m', this_exchange.parse8601(self.options.use_start_date)
        )

        # put that in the custom dataframe
        df = DataStruct(OHLCVS,
                        columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        df.exchange = exchange
        coin1, coin2 = pair.split('/')
        df.pair = '{}_{}'.format(coin1, coin2)
        return df

    def get_playback_candles(self, exchange='poloniex', pair='BTC/USDT'):
        """
        :param exchange: this should be known in advance with api information in the config.
        :param pair: A known trading pair, try seaching for this first and insert programatically.
        :return: A custom pandas dataframe with a fix_time and superset functions added.
        """
        # add setup info for ccxt
        config = {'rateLimit': 3000,
                  'enableRateLimit': True,
                  # 'verbose': True,
                  }
        # this is a setup with no login.
        this_exchange = eval('ccxt.{}({})'.format(exchange, config))
        time.sleep(this_exchange.rateLimit / 1000)

        # this is the webcall for candlestick data
        OHLCVS = this_exchange.fetch_ohlcv(
            pair, '5m', (int(time.time())-7200)*1000
        )

        # put that in the custom dataframe
        df = DataStruct(OHLCVS,
                        columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        df.exchange = exchange
        coin1, coin2 = pair.split('/')
        df.pair = '{}_{}'.format(coin1, coin2)
        return df
