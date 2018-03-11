#!/usr/bin/python3
"""
Bittensor.
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
import os, sys, time, datetime, collections, re, random, asyncio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from timeit import default_timer as timer
runtime = timer()
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler

import ag.bittensor.utils.options as options
import ag.bittensor.ai.make_data as make_data
from ag.bittensor.ai.AI import Q_Trader
import ag.bittensor.utils.talib as talib

class Bittensor(object):
    """
    Bittensor.
    Another AlphaGriffin Project 2018.
    Alphagriffin.com
    """

    def __init__(self, options):
        """Use the options for a proper setup."""
        self.options = options
        self.model = Q_Trader(options)
        self.datasmith = make_data.MakeData(options)
        self.TA = talib.TALib()

        # DATAFRAME OPTIONS
        self.timeframe = '1H'  # T for mins, H, D, AN for year.
        self.period = 12

        # training Options
        self.sample_size = 80

        ## WTF ERRORS
        np.seterr(all='ignore')

    def main(self):
        print('Starting BitTensor')
        sample_file = self.datasmith.random_filename
        self.datasmith.dataframe = sample_file
        print('Calculating Sample Data to Period: {}, TimeFrame: {}'.format(
            self.period, self.timeframe))
        df, num_df, _ = self.make_sample(sample_file, self.period,self.timeframe)
        print('Columns in dataset:\n\t', '\n\t'.join([x for x in df.columns]))
        self.model.set_state_dim(
            num_df[0].shape[0]
        )
        self.model.preRun()
        print('Finished Setup. Starting Training')
        start = timer()
        while True:
            self.model.reset_que()
            # sample_file = self.datasmith.random_filename
            sample_file = self.datasmith.next_filename
            if sample_file is None: break;
            # ONLY GET BTC PAIRS

            ## DO A VOLUME SORT
              # if volume sucks: dont train on that shit
            ## /volumesort


            df, num_df, samples = self.make_sample(sample_file, self.period, self.timeframe)

            # if self.datasmith.candles['Volume'].iloc[-1] < 50:
            #     # print('{} has low volume... skipping'.format(sample_file))
            #     continue

            if num_df.shape[0] > 200:
                print('WTF: {}'.format(sample_file))
                continue

            print('Training file {}'.format(sample_file))
            last_sample = None
            for index, sample in enumerate(samples):
                sample = sample.reshape(1, -1)
                if last_sample is None:
                    last_sample = sample
                    continue
                action = self.model.egreedy_action(last_sample)
                reward, _ = self.model.get_reward(action, index, df)
                # print('Action: {}, Reward: {}'.format(action, reward))
                # TRAIN THE MODEL!
                self.model.train(last_sample, action, reward, sample)


                # if index % 100 == 0:
                #     print('finished {}/{} iters'.format(index, num_df.shape[0]))
                #      print('Recap: steps {} | loss {}'.format(
                #         self.model.stats.g_step, self.model.stats.cost
                #     ))
                # break
            self.model.save_or_load()
            took = timer() - start
            print('Current Runtime {:.2f}secs'.format(took))
            print('Recap: steps {} | loss {}'.format(
                self.model.stats.g_step, self.model.stats.cost
            ))
            # break
        print('Finished Training')
        return True

    def make_sample(self, sample, period, timeframe):
        def unzero(df):
            df = df.replace(0, 'NaN')
            df = df.dropna(how='all', axis=0)
            df = df.replace('NaN', 0)
            df.len = len(df)
            return df

        # make data
        self.datasmith.dataframe = sample
        self.datasmith.candles = timeframe
        df = self.datasmith.candles
        # get all the TA
        B1, MA, B2 = self.TA.BBANDS(df, period)
        EMA = self.TA.EMA(df, period + 1)
        RSI = unzero(self.TA.RSI(df, period))
        MOM = unzero(self.TA.MOM(df, period - 3))
        CCI = self.TA.CCI(df, period)
        # concatenate with RSI first because it has no x axis
        input_df = pd.DataFrame({
            'RSI': RSI.values,
        }, index=MOM.index[:len(RSI)])
        for i in [df['Close'], B1, MA, B2, EMA, MOM, CCI]:
            input_df = input_df.join(i)
        # fill gaps
        input_df_ = input_df.fillna(method='bfill')
        # make NP array
        input_df = np.array(input_df_)
        # normalize
        if False:
            input_df = np.diff(input_df, axis=0) / input_df[1:] * 100
        # pull infinity
        if True:
            for i in range(len(input_df)):
                for k, j in enumerate(input_df[i]):
                    if not np.isfinite(j):
                        input_df[i][k] = 0
        # make polynomially complex( add complexity for no good reason )
        if False:
            poly = PolynomialFeatures(degree=4, interaction_only=True)
            input_df = poly.fit_transform(input_df)
        # scale it from -1 to 1
        if True:
            scaler = MaxAbsScaler()
            input_df = scaler.fit_transform(input_df)
        # return pandas DF, Numpy DF, and the iterator of Numpy
        # return the complete dataset for index reference
        # AND an iterator so that you know when your done
        return input_df_, input_df, iter(input_df)

def main():
    """Loads Options ahead of the app"""
    config = options.Options('config/access_codes.yaml')
    app = Bittensor(config)
    try:
        app.main()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
    # os.system('cls')
    print('Thanks!')
    print('BitTensor - AlphaGriffin | 2018')
