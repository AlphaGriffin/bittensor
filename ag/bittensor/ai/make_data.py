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

# generic
import os, sys, time, datetime, collections, re, random
from itertools import cycle
import ag.bittensor.utils.options as options
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
        self.file_loc = os.path.join(os.getcwd(), 'data', 'files')
        self.columns = [
                'timestamp', 'high', 'low',
                'last', 'change', 'baseVolume'
                ]

        self._dataframe = None
        self._candleframe = None
        self._pair = None
        self.max_time_frame = 75
        self._next_filename = None

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
            self.next_filename = self.file_loc
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
        self._candleframe = self.make_candles(value)

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
        self._candleframe = self.make_candles(self.dataframe)
        self._dataframe.pair = self._candleframe.pair = value[:-4]
        
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

        seven_min_change = unzero(dataframe.pct_change(  periods = 7))
        twelve_min_change = unzero(dataframe.pct_change( periods = 12))
        twenty_min_change = unzero(dataframe.pct_change( periods = 20))
        forty_min_change = unzero(dataframe.pct_change(  periods = 40))
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
            X3 = pd.Series(_set['last']).astype(float).as_matrix()
            # X4 = pd.Series(_set['change']).astype(float).as_matrix()
            X5 = pd.Series(_set['baseVolume']).astype(float).as_matrix()
            X6 = np.append([X1, X2, X3], X5)  # remove x4
            inputs.append(X6)
        return inputs

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

    def make_candles(self, df, column='last', period='5T'):
        '''Slice the data for any candle periods'''
        candles = pd.DataFrame()
        candles['Open'] = df[column].resample(period).first().values
        candles['High'] = df[column].resample(period).max().values
        candles['Low'] = df[column].resample(period).min().values
        candles['Close'] = df[column].resample(period).last().values
        candles['timestamp'] = pd.DatetimeIndex( 
            df['timestamp'][:len(candles)].values
            )
        candles.set_index('timestamp', inplace=True)
        return candles

        

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
