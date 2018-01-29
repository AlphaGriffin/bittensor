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
# from ag.bittensor.ai.AI import Q_Trader as bot

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
        self.robot = bot()
        log.debug('Loaded Agent')

        # CONSOLE LOGGING
        self.P('Setup Complete')
        log.debug('setup complete')

        self.P('Starting BitTensor Service.')
        log.debug('Checking Configuration...')
        self.P('Using Exchanges:')
        for id in list(self.options.use_exchanges.split(',')):
            self.P("{}".format(id))
        self.P('Trading Pairs:')
        for id in list(self.options.use_pairs.split(',')):
            self.P("{}".format(id))
        pass

    async def get_ticker(self, exchange='poloniex', symbol='LTC/BTC'):
        this_exchange = eval('ccxt.{}()'.format(exchange))
        return await this_exchange.fetch_ticker(symbol)

    # @setInterval(300)
    def predict(self, exchange='poloniex', symbol='LTC/BTC'):
        p_start = timer()
        df = self.dataHandler.get_playback_candles(exchange, symbol)
        df = df.mircoset()
        action = self.Agent.egreedy_action(df)
        if action == 0:
            print('BUY!')
        elif action == 1:
            print('SELL!')
        else:
            print('DO NOTHING.')
        print('this action took {} secs to execute.'.format(timer()-p_start))

    def main(self):
        print('This is working!')
        self.predict()
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
