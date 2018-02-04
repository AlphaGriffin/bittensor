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
from timeit import default_timer as timer
from decimal import Decimal as D
import pandas as pd
import numpy as np

import ccxt

from tqdm import tqdm, trange

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
        pair = df.pair
        cols = []
        for i in range(20):
            cols.append('close_{}'.format(i))
            cols.append('vol_{}'.format(i))

        # working_df = pd.DataFrame(columns=cols)
        working_df = DataStruct(columns=cols)
        working_df.pair = pair
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
        pair = df.pair
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
        # adding in func save feature
        filename = 'df_{}_{}.csv'.format(exchange, pair)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if os.path.exists(os.path.join(dir_path, 'data', 'datasets', filename)):
            self.P('Loading Saved for {} | {}'.format(exchange, pair))
            df = pd.read_csv(os.path.join(dir_path, 'data', 'datasets', filename))
            coin1, coin2 = pair.split('/')
            df.pair = '{}_{}'.format(coin1, coin2)
            df.exchange = exchange
        else:
            config = {'rateLimit': 3000,
                      'enableRateLimit': True,
                      # 'verbose': True,
                      }
            # this is a setup with no login.
            this_exchange = eval('ccxt.{}({})'.format(exchange, config))
            start_time = timer()
            time.sleep(this_exchange.rateLimit / 400)

            # this is the webcall for candlestick data
            OHLCVS = this_exchange.fetch_ohlcv(
                pair, '5m', this_exchange.parse8601(self.options.use_start_date)
            )
            print('Downloaded {}'.format(pair))
            # put that in the custom dataframe
            df = DataStruct(OHLCVS,
                            columns=['time', 'open', 'high', 'low', 'close', 'vol'])
            df.exchange = exchange
            coin1, coin2 = pair.split('/')
            df.pair = '{}_{}'.format(coin1, coin2)
            df.to_csv(os.path.join(dir_path, 'data', 'datasets', filename))
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

    def pack_rows3(self, pairs, samples, count, stop=None):
        size = count - (samples - 1)
        if stop is not None and stop < size:
            size = stop

        result = np.empty(shape=(size, len(pairs)*2*samples)) # rows of 1080 columns each (9 pairs * 2 variables * 60 historical samples)

        for p, rows in enumerate(pairs):
            # print(rows.pair)
            history = [] # keep 'samples' historical entries
            r = -1
            for rr, row in rows.iterrows():
                r += 1
                if stop is not None and r >= stop + samples - 1:
                    break
                history.append(row)

                if r < samples - 1:
                    continue

                columns = result[r - (samples - 1)]

                for s in range(samples):
                    entry = history[samples - 1 - s]
                    columns[p*samples+s*2] = entry['close']
                    columns[p*samples+s*2+1] = entry['vol']

                history.pop(0) # keep history trimmed to 'samples' size

        return result

    def get_all_dataframes(self):
        coins_to_use = ['ETH/BTC', 'XRP/BTC', 'DASH/BTC' , 'XMR/BTC',
                'BTS/BTC', 'DOGE/BTC', 'FCT/BTC', 'MAID/BTC', 'CLAM/BTC']
        datasets = []
        min_len = 1000000
        # download all the coins
        print('Downloading coins')
        for coin in coins_to_use:
            chart = self.get_candles(pair=coin)
            # time.sleep(2)
            this_len = len(chart)
            if this_len < min_len:
                min_len = this_len
            datasets.append(chart)
        print('Downloaded {} coins.'.format(len(datasets)))
        new_datasets = []
        # trim all the datasets to min_len
        print('Trimming Lens')
        for d in datasets:
            pair = d.pair
            this_len = len(d)
            if this_len == min_len:
                new_datasets.append(d)
            elif this_len > min_len:
                trim_section_start = len(d['time']) - min_len
                df = d[trim_section_start:]
                df.pair = pair
                new_datasets.append(df)
        superset = []
        elements_per_coin = 20
        num_coins = len(new_datasets)
        my_range = min_len - elements_per_coin
        print('Creating {} Rows'.format(my_range))
        for i in trange(my_range):
        # for i in range(3333):
            if i % 1000 == 0:
                tqdm.write('Completed {} of {} rows.'.format(i, my_range), end='\r')
            if i > elements_per_coin:
                datarow = []
                for d in range(num_coins):
                    dataframe = new_datasets[d]
                    _len = len(dataframe)
                    curr = _len - i
                    for j in range(elements_per_coin):
                        price = D(dataframe.loc[curr]['close'])
                        vol = D(dataframe.loc[curr]['vol'])
                        datarow.append(price)
                        datarow.append(vol)
                        # do mean for last 20 elements... i think its just df[x-20:x].mean()
                superset.append(datarow)

        # for df in new_datasets:
        #     pair = df.pair

        # superset = self.pack_rows3(new_datasets, 60, min_len, 1000)
        return np.array(superset)

    def get_apples(self, exchange='bitmex'):
        pass
