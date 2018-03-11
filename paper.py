#!/usr/bin/python3
"""
Bittensor Decider.
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
import numpy as np
import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from timeit import default_timer as timer
runtime = timer()
from tqdm import tqdm, trange

# crypto
import ccxt
import asyncio
import ccxt.async as cryptosync

# Tensorflow
from ag.bittensor.ai.AI import Q_Trader as bot

# Dataset
from ag.bittensor.ai.make_data import MakeData
import pandas as pd

# GameEngine // need access to engine for reward data
import ag.bittensor.game.stats as stats

# Utilities
import ag.bittensor.utils.options as options
import ag.bittensor.utils.plotter as plotter
# import ag.bittensor.utils.sheets as sheets
import ag.bittensor.utils.slack as slack
import ag.bittensor.utils.talib as talib
import ag.bittensor.utils.printer as printer
# from ag.bittensor.strategy.macd import *
# import ag.logging as log

# Game Stuff
import ag.bittensor.game.stats as stats
# log.set(log.WARN)

class Signals(object):

    @staticmethod
    def MACD_signals(Smith, slow, fast, period='15m'):
        Smith.candles = period
        candles = Smith.candles
        candles['slow'] = np.round(candles["Close"].rolling(window=slow, center=False).mean(), 8)
        candles['fast'] = np.round(candles["Close"].rolling(window=fast, center=False).mean(), 8)
        candles['macd'] = candles['fast'] - candles['slow']

        ## WHY USE CANDLE PRICE HERE?? SEEMS LIKE MADNESS...
        # because this - that = positive is not in full dollar units... so the threshhold needs to
        # be multipiled by the incoming coin price for the right amount of decimal places to be ahead or
        # behind by. the value has to change 1/100 of the close price or it wont signal.
        # now that i think about it this threshold could be in the options... i guess.
        candles['macd_regime'] = np.where(candles['macd'] > candles['Close']*.01, 1, 0)
        candles['macd_regime'] = np.where(candles['macd'] < -candles['Close']*.01, -1, candles['macd_regime'])
        candles['macd_signal'] = candles['macd_regime'] - candles['macd_regime'].shift(1)
        # candles.sort_index(inplace=True)
        # candles['macd_signal'].plot().axhline(y = 0, color = "black", lw = 2)
        return candles

    @staticmethod
    def momentum_signals(Smith, mom, period='15m'):
        TA = talib.TALib()
        Smith.candles = period
        candles = Smith.candles
        x = TA.MOM(candles, mom, 'Close')
        x['Momentum_regime'] = np.where(x['Momentum_Close_{}'.format(mom)] > x['Close']*0.01, 1, 0)
        x['Momentum_regime'] = np.where(x['Momentum_Close_{}'.format(mom)] < -x['Close']*0.01, -1, x['Momentum_regime'])
        x['Momentum_signal'] = x['Momentum_regime'] - x['Momentum_regime'].shift(1)
        return x


class Engine(object):
    def __init__(self, options):
        # set globals
        self.options = options
        self.P = printer.Printer(options)
        self.slacker = slack.Slacker(options)
        self.game = stats.Stats(options)
        self.signals = Signals()
        self.datasmith = MakeData(options)
        self.reset_game_options()
        self.reset_feedback()
        self.P('Starting FauxTrader')
        # self.slacker.Print('Decider Bot is coming online Now.')

    def reset_game_options(self):
        # moving averages
        self.slow_period = 42
        self.fast_period = 21
        # momentum ... int works... does a timeref???
        self.mom_period = 12
        # time frame for candles... int works so does timeref... ie. 5T... T for mins
        self.time_frame = '1H'
        # Sorting the volume feels right.... High, Low, Banded, None
        self.volume_sort = None
        # if volume sort then use band high/low
        self.vol_band_high = 333
        self.vol_band_low = 13

    def reset_feedback(self):
        self.top_pairs = pd.DataFrame(columns=['Pair', 'Profits'])
        self.winners = []
        self.losers = []
        self.all_total_trades = 0
        self.winner_returns = []
        self.losers_returns = []
        self.candles = None
        self.theReturn = 0
        self.all_losing_returns = 0
        self.all_winners_returns = 0
        self.all_fees_paid = 0
        self.start_cost = 0
        self.start_time = None
        self.end_time = None

    def main(self, signal=['mom']):
        start = timer()
        # process all coins.
        for i in trange(self.datasmith.total_coins):
            filename = self.datasmith.next_filename
            if filename is None: break
            ## SORT 1: ONLY BITCOIN PAIRS
            if not 'BTC' in filename[:-4][-3:]:
                continue
            self.datasmith.dataframe = filename
            ## SORT 2: ONLY HIGH VOLUME PAIRS
            self.datasmith.candles = self.time_frame
            if self.volume_sort:
                if 'High' in self.volume_sort:
                    if not self.datasmith.candles['Volume'].iloc[-1] >= 100:
                        continue
                elif 'Low' in self.volume_sort:
                    ## SORT 2a: ONLY LOW VOLUME PAIRS
                    if not self.datasmith.candles['Volume'].iloc[-1] <= 100:
                        continue
                elif 'banded' in self.volume_sort:
                    ## SORT 2a: ONLY LOW VOLUME PAIRS
                    if self.datasmith.candles['Volume'].iloc[-1] <= self.vol_band_low:
                        continue
                    if self.datasmith.candles['Volume'].iloc[-1] >= self.vol_band_high:
                        continue

            """ Print the progress
            tqdm.write('Processing Coin: {}, Volume: {:.2f}, signal: {}'.format(
                filename[:-4], self.datasmith.candles['Volume'].iloc[-1],
                'MACD_{}_{}_{}'.format(self.time_frame, self.slow_period, self.fast_period)
            ))
            """
            # this signal should be PASSED IN
            if signal:
                for i in signal:
                    if 'macd' in i:
                        self.candles = self.signals.MACD_signals(
                            self.datasmith,
                            slow=self.slow_period,
                            fast=self.fast_period,
                            period=self.time_frame
                        )
                    if 'mom' in i:
                        self.candles = self.signals.momentum_signals(
                            self.datasmith,
                            mom=self.mom_period,
                            period = self.time_frame
                        )


            # THIS PAIR OF THINGS RESETS AFTER EACH COIN... this is good.
            self.game.reset_paperTrader()
            profits = self.game.process_trades(self.candles)
            self.all_fees_paid += self.game.all_fees_paid



            # THIS IS NOT PYTHONIC
            if profits > 0:
                self.winners.append([filename, profits, len(self.game.paperTrader)])
                self.top_pairs = self.top_pairs.append({
                    "Pair": filename[:-4],
                    "Profits": profits
                    },
                    ignore_index=True)
            else:
                self.losers.append([filename, profits, len(self.game.paperTrader)])
            # break

        # SET TIME PERIOD CHECKED
        self.start_time = self.datasmith.dataframe.start_date
        self.end_time = self.datasmith.dataframe.end_date

        self.top_pairs.set_index('Profits')
        self.top_pairs.sort_values('Profits', ascending=False)
        # print some results
        print('Took {:.2f} secs'.format(timer()-start))
        self.get_results()
        self.print_results('term')
        # self.print_results('slack')

    def get_results(self):
        for i in sorted(self.winners):
            if i[1] > 0:  # scratch nan
                self.winner_returns.append(i[1])
                self.all_total_trades += i[2]
        for i in sorted(self.losers):
            if i[1] < 0:  # scratch nan
                self.losers_returns.append(i[1])
                self.all_total_trades += i[2]
        self.start_cost = 0.001 * (len(self.winners) + len(self.losers))
        self.all_winners_returns = sum(self.winner_returns)
        self.all_losing_returns = sum(self.losers_returns)
        self.theReturn = self.start_cost + self.start_cost*(self.all_winners_returns + self.all_losing_returns)

    def print_results(self, target='term'):
        msgs = []
        msgs.append( '_.|::_    *BitTensor BackTest Report*    _::|._' )
        msgs.append( '*Signal(s)*: MACD, s:{}, f:{}, t:{}'.format(self.slow_period, self.fast_period, self.time_frame) )
        msgs.append( '*Time Period*:\n\ts: {}\n\te: {}'.format(self.start_time, self.end_time) )
        msgs.append( '*Volume Sort*: {}'.format(self.volume_sort) )
        if self.volume_sort:
            if 'banded' in self.volume_sort:
                msgs.append( '24H Volume *High* Limit: {}'.format(self.vol_band_high) )
                msgs.append( '24H Volume *Low* Limit: {}'.format(self.vol_band_low) )
            msgs.append( '*Number of Coins Winning/Losing*: {}/{}'.format(
                len(self.winners), len(self.losers)
            ) )
        if len(self.top_pairs) > 3:
            msgs.append( 'Top Pairs:\n\t*{}* + {:.2f}%\n\t*{}* + {:.2f}%\n\t*{}* + {:.2f}%'.format(
                self.top_pairs['Pair'].iloc[0], self.top_pairs['Profits'].iloc[0]*100,
                self.top_pairs['Pair'].iloc[1], self.top_pairs['Profits'].iloc[1]*100,
                self.top_pairs['Pair'].iloc[2], self.top_pairs['Profits'].iloc[2]*100,
            ))
            msgs.append( 'Signal Produced _{:.2f}%_ trading all pairs in {} trades with {:.8f}b in total fees.'.format(
                (self.all_winners_returns+self.all_losing_returns)*100,
                self.all_total_trades,
                self.game.all_fees_paid
            ) )
            msgs.append('Total Cost/Return to trade all _{}_ pairs\n*Exchange {}*: _{:.4f}b_ / _{:.4f}b_'.format(
                len(self.winners)+len(self.losers), 'Bittrex', self.start_cost, self.theReturn
            ))
        else:
            msgs.append('*This Strategy Sucks.*')
        msgs.append('`END OF REPORT`')

        if 'term' in target:
            PRINTER = print
        elif 'slack' in target:
            PRINTER = self.slacker.Print
        elif 'pretty' in target:
            PRINTER = self.P
        else:
            # PRINTER = log.debug
            pass

        PRINTER('\n'.join([x for x in msgs]))
        # NOT PYTHONIC
        #for i in msgs:
        #    PRINTER(str(i))
        #    time.sleep(.25)
        return True


def main():
    """Loads Options ahead of the app"""
    config = options.Options('config/access_codes.yaml')
    signal_ = 'mom'
    app = Engine(config)
    try:
        app.main(signal_)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
    # os.system('cls')
    print('Thanks!')
    print('BitTensor - AlphaGriffin | 2018')
