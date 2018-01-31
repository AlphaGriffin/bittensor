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
import ag.bittensor.ai.dataprep as dataprep
from ag.bittensor.ai.dataprep import DataStruct
import pandas as pd

# GameEngine // need access to engine for reward data
import ag.bittensor.game.engine as engine

# Utilities
import ag.bittensor.utils.options as options
import ag.bittensor.utils.plotter as plotter
# import ag.bittensor.utils.sheets as sheets
import ag.bittensor.utils.printer as printer
import ag.logging as log

# set globals
log.set(log.WARN)


def setInterval(interval):
    def decorator(function):
        def wrapper(*args, **kwargs):
            stopped = threading.Event()

            def loop(): # executed in another thread
                while not stopped.wait(interval): # until stopped
                    function(*args, **kwargs)

            t = threading.Thread(target=loop)
            t.daemon = True # stop if the program exits
            t.start()
            return stopped
        return wrapper
    return decorator


class BitTensorPlay(object):
    def __init__(self, options):
        self.options = options
        # build objects
        self.plotter = plotter.Plot(options)
        log.debug('Loaded Plotter Program.')
        # self.sheets = sheets.gHooks(options)
        # log.debug('Loaded gHooks Program.')
        self.P = printer.Printer(options)
        log.debug('Loaded Printer Program.')
        # self.agent = autoI.Q_Trader(options)
        # log.debug('Loaded AI Program.')
        self.dataHandler = dataprep.DataHandler(options)
        log.debug('Loaded Data Handler Program')
        self.gameEngine = engine.GameEngine(options)
        log.debug('Loaded Game Engine.')
        self.robot = bot(options)
        log.debug('Loaded Agent')

        # CONSOLE LOGGING
        self.P('Setup Complete')
        log.debug('setup complete')

        self.P('Starting BitTensor Service.')
        log.debug('Checking Configuration...')
        self.P('Using Exchanges:')

        for id in list(self.options.use_exchanges.split(',')):
            self.P("{}".format(id))
        # self.P('Trading Pairs:')
        self.trading_pairs = []
        for id in list(self.options.use_pairs.split(',')):
            # self.P("{}".format(id))
            self.trading_pairs.append(id)
        pass

    async def get_ticker(self, exchange='poloniex', symbol='LTC/BTC'):
        this_exchange = eval('ccxt.{}()'.format(exchange))
        return await this_exchange.fetch_ticker(symbol)

    # @setInterval(300)
    def predict(self,exchange='poloniex', symbol='LTC/BTC',  dataframe=None):
        p_start = timer()
        if dataframe is None:
            df = self.dataHandler.get_playback_candles(exchange, symbol)
        else:
            df = dataframe
        df = df.mircoset()
        action = self.robot.egreedy_action(df)
        # print('this action took {} secs to execute.'.format(timer()-p_start))
        return action

    def main(self):
        self.P('Starting BitTensorPlay.')
        # self.P('Getting Data for pairs')
        datasets = []
        for i in self.trading_pairs:
            self.P('Got data for {}'.format(i))
            datasets.append(self.dataHandler.get_candles(pair=i))
            break
            time.sleep(3)
        self.P('Downloaded all of the datasets.')

        # do the predictions on each dataset with the same model.
        for i in datasets:
            # i is our df for the pair
            pair = i.pair
            len_of_set = len(i)
            index = 0
            action_set = []
            tt = i.loc[0]['time']
            #print(tt)
            #starting_time = datetime.datetime.utcfromtimestamp(1509898800000.0).strftime('%Y-%m-%d %H:%M:%S')
            starting_btc = 1
            current_btc_bal = starting_btc
            current_coin_bal = 0
            start_coin_time = timer()
            last_buy = 0
            first_price = 0
            has_bought = False
            last_buy_price = 0
            winning_trades = 0
            losing_trades = 0
            for x in trange(len(i) - 20,
                            ascii=True,
                            desc="{} Historical since {}".format(pair, tt),
                            dynamic_ncols=True,
                            smoothing=0,
                            leave=False,
                            unit_scale=True
                            ):

                df = i[index:x+20]
                action = self.predict(dataframe=df)
                action_set.append(action)
                current_price = i.loc[x+20]['close'] ## GOT TO HERE
                current_time = i.loc[x+20]['time']
                if x == 0:
                    first_price = current_price
                if action == 0:
                    if not has_bought:
                        tqdm.write('BUY! @ {}'.format(current_price))
                        current_btc_bal -= current_price
                        current_coin_bal += 1
                        last_buy_price = current_price
                        has_bought = True
                elif action == 1:
                    if has_bought:
                        tqdm.write('SELL! @ {}'.format(current_price))
                        current_btc_bal += current_price
                        current_coin_bal -= 1
                        if current_price > last_buy_price:
                            winning_trades += 1
                            tqdm.write('WIN! total Wins: {} total Loses: {}'.format(
                                winning_trades, losing_trades
                            ))
                        else:
                            losing_trades += 1
                        last_buy_price = 0
                        has_bought = False

                else:
                    tqdm.write('DO NOTHING.')
                    pass

                index += 1

            last_price = current_price
            buy_hold = last_price / first_price
            sell_off_bal = current_coin_bal * current_price
            current_btc_bal += sell_off_bal
            self.P('Exchange: {}, Pair: {}, Final_BTC: {:.8f}, B/H: {:.2f}, P/L: {:.2f}%, S/B: {:.2f}%'.format(
                'poloniex', pair,
                D(current_btc_bal),
                D(buy_hold),
                D(current_btc_bal/starting_btc) - 1,
                pd.Series(action_set).mean()
            ))
            self.P('Backtesting this coin took {:.2f} mins'.format(D(timer()-start_coin_time)/60))
        print('is this thing still on?')
        return True


def main():
    """Launcher for the app."""
    if os.path.exists('config/access_codes.yaml'):
        config = options.Options('config/access_codes.yaml')
    else:
        print('\n#| AlphaGriffin - BitTensor |#')
        print(": To begin copy the dummy_codes.yaml file,")
        print(": the one thats in the config folder in this repo.")
        print(': to access_codes.yaml.\n')
        print(": After that, restart this app.")
        exit('AlphaGriffin | 2018')
    app = BitTensorPlay(config)
    if app.main():
        return True
    return False

if __name__ == '__main__':
    try:
        if main():
            print("AlphaGriffin  |  2018")
        else:
            print("Controlled Errors. Good day.")
    except Exception as e:
        print("and thats okay too.")
        log.error(e)
        sys.exit()
