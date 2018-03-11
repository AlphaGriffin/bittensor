#!/usr/bin/python3
"""
Bittensor. Datasmith.
by: AlphaGriffin
"""

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2018, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.3"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Beta"

# generic
import os, sys, time, datetime, collections, re, random
from itertools import cycle
import numpy as np
import pandas as pd

class MakeData(object):
    """
    Bittensor.
    Another AlphaGriffin Project 2018.
    Alphagriffin.com
    """

    def __init__(self, options=None):
        """Use the options for a proper setup."""
        self.options = options
        self._dir_loc = os.path.join(os.getcwd(), 'exchanges')
        # self.file_loc = 'C:\\datasets\\files' # fucking windows
        self.columns = ['timestamp','Open','High','Low','Close','Volume']
        self.file_loc = ''
        self._cur_exchange = ''
        self._all_exchanges = []
        self._all_pairs = []
        self._dataframe = None
        self._candleframe = None
        self._pair = None
        self.max_time_frame = 75
        self._next_filename = None

        self.all_exchanges = self._dir_loc
        self.cur_exchange = self.all_exchanges[0]
        self.next_filename = self.file_loc

    def main(self):
        sample = self.random_filename
        print('{}'.format(self.pair))

        self.dataframe = sample
        print(self.dataframe.tail(2))

        normal = self.make_normal(self.dataframe)
        inputs = self.make_input_from_normal(normal)
        for i in inputs:
            print(i)
        return True

    @property
    def cur_exchange(self):
        return self._cur_exchange

    @cur_exchange.setter
    def cur_exchange(self, value):
        self._cur_exchange = value
        ex_dir = os.path.join(self._dir_loc, value)
        self.file_loc = ex_dir
        self.all_pairs = ex_dir
        self.total_coins = len(os.listdir(ex_dir))

    @property
    def pair(self):
        return self._pair

    @pair.setter
    def pair(self, value):
        # value = '{}_{}'.format(value.split('/')[0], value.split('/')[1])
        value = value[:-4]  # remove .csv
        self._pair = value

    @property
    def random_filename(self):
        filename = os.listdir(self.file_loc)[random.randint(1, len(os.listdir(self.file_loc)))]
        self.pair = filename
        return filename

    @property
    def next_filename(self):
        try:
            filename = next(self._next_filename)
            self.pair = filename
        except:
            self._next_filename = iter(os.listdir(self.file_loc))
            filename = None
        return filename

    @next_filename.setter
    def next_filename(self, value):
        self._next_filename = iter(os.listdir(value))

    @property
    def candles(self):
        return self._candleframe

    @candles.setter
    def candles(self, value):
        # maybe this could take a tuple of (df, periods)
        # but that doesnt sound intuitive.. ???
        self._candleframe = self.make_candles(period=value)
        self._candleframe.pair = self.pair

    @property
    def dataframe(self):
        return self._dataframe

    @dataframe.setter
    def dataframe(self, value):
        self._dataframe = pd.read_csv(
                os.path.join(self.file_loc, value),
                names=self.columns
        )
        self._dataframe = self.fix_time(self._dataframe)
        self._dataframe.set_index(
            pd.DatetimeIndex(
                self._dataframe['timestamp']),
            inplace=True)
        self._dataframe.start_date = self._dataframe.index[0]
        self._dataframe.end_date = self._dataframe.index[-1]
        self._candleframe = self.make_candles(self.dataframe)
        self._dataframe.pair = self._candleframe.pair = value[:-4]

    @property
    def all_pairs(self):
        return self._all_pairs

    @all_pairs.setter
    def all_pairs(self, value):
        self._all_pairs = [x for x in os.listdir(value)]

    @property
    def all_exchanges(self):
        return self._all_exchanges

    @all_exchanges.setter
    def all_exchanges(self, value):
        # list(os.listdir(value))
        self._all_exchanges = [x for x in os.listdir(value)]


    """ DEPRICATED!!
    def make_normal(self, dataframe=None):
        if dataframe is None:
            dataframe = self.dataframe
        # dataframe = dataframe.tail(self.max_time_frame)
        date_col = pd.Series(dataframe.pop('timestamp'))

        def unzero(df):
            df = df.replace(0,'NaN')
            df = df.dropna(how='all',axis=0)
            df = df.replace('NaN', 0)
            df.len = len(df)
            return df

        seven_min_change = unzero(dataframe.pct_change(  periods = 7 ))
        twelve_min_change = unzero(dataframe.pct_change( periods = 12 ))
        twenty_min_change = unzero(dataframe.pct_change( periods = 20 ))
        forty_min_change = unzero(dataframe.pct_change(  periods = 40 ))
        return [dataframe,
                seven_min_change,
                twelve_min_change,
                twenty_min_change,
                forty_min_change,
                date_col]

    def make_input_from_normal(self, datasets):
        normal_sets = datasets[1:-1]
        inputs = []
        for i in normal_sets:
            _set = i
            X1 = pd.Series(_set['high']).astype(float).as_matrix()
            X2 = pd.Series(_set['low']).astype(float).as_matrix()
            X3 = pd.Series(_set['Close']).astype(float).as_matrix()
            # X4 = pd.Series(_set['change']).astype(float).as_matrix()
            X5 = pd.Series(_set['Volume']).astype(float).as_matrix()
            X6 = np.append([X1, X2, X3], X5)  # remove x4
            inputs.append(X6)
        return inputs
    """

    def fix_time(self, dataframe=None):
        if dataframe is None:
            dataframe = self.dataframe
        time = dataframe.pop('timestamp')
        transform = time.apply(lambda x:
                                datetime.datetime.fromtimestamp(
                                        x/1000
                                    ).strftime('%Y-%m-%d %H:%M:%S'))
        dataframe = dataframe.join(transform, how='inner')
        return dataframe

    def make_candles(self, df=None, column='Close', period='5T'):
        '''Slice the data for any candle periods'''
        if df is None:
            df = self._dataframe
        candles = pd.DataFrame()
        candles['Open'] = df['Open'].resample(period).first().values
        candles['High'] = df['High'].resample(period).max().values
        candles['Low'] = df['Low'].resample(period).min().values
        candles['Close'] = df['Close'].resample(period).last().values
        candles['Volume'] = df['Volume'].resample(period).last().values
        try:
            candles['timestamp'] = pd.DatetimeIndex(
                # this is not working properly !!
                df['timestamp'][:len(candles.index)].values
                # df['timestamp'][:].values
                )
        except:
            candles['timestamp'] = pd.DatetimeIndex(
                # this is not working properly !!
                df['timestamp'].values
                )
        candles.fillna(method='bfill')
        candles.set_index('timestamp', inplace=True)

        return candles


class CoinData(object):
    def __init__(self):
        pass

def main():
    """Loads Options ahead of the app"""
    config = options.Options('config/access_codes.yaml')
    app = MakeData(config)
    try:
        app.main()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
    # os.system('cls')
    print('Thanks!')
    print('BitTensor - AlphaGriffin | 2018')
