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
from timeit import default_timer as timer
from tqdm import tqdm, trange

# crypto
import ccxt

# Tensorflow
import ag.bittensor.ai.AI as autoI

# Dataset
import ag.bittensor.ai.dataprep as dataprep
from ag.bittensor.ai.dataprep import DataStruct
import pandas as pd

# GameEngine // need access to engine for reward data
import ag.bittensor.game.engine as engine

# REPLACING THE OLD STUFFS
from ag.bittensor.ai.stock_env import StockEnv
from ag.bittensor.ai.DQN_trade import DQN_Trade as bot

# Utilities
import ag.bittensor.utils.options as options
import ag.bittensor.utils.plotter as plotter
# import ag.bittensor.utils.sheets as sheets
import ag.bittensor.utils.printer as printer
import ag.logging as log

# set globals
log.set(log.WARN)


class Bittensor(object):
    """
    Bittensor.
    Another AlphaGriffin Project 2018.
    Alphagriffin.com
    """

    def __init__(self, options):
        """Use the options for a proper setup."""
        self.options = options
        self.options.save_file = True

        # build objects
        self.plotter = plotter.Plot(options)
        log.debug('Loaded Plotter Program.')
        # self.sheets = sheets.gHooks(options)
        log.debug('Loaded gHooks Program.')
        self.P = printer.Printer(options)
        log.debug('Loaded Printer Program.')
        self.agent = autoI.DQN_Trader(options)
        log.debug('Loaded AI Program.')
        self.dataHandler = dataprep.DataHandler(options)
        log.debug('Loaded Data Handler Program')
        self.gameEngine = engine.GameEngine(options)
        log.debug('Loaded Game Engine.')

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

    def main(self):
        exchange = 'poloniex'
        pair = 'LTC/BTC'
        runtime = timer()
        ss = self.get_dataset(exchange, pair)
        self.P('Total Runtime to collect and process all data for 1 pair is {:.1f} mins'.format(float(timer()-runtime)/60))
        # start AI engine.
        Agent = bot()
        data = ss
        if True:
            self.P('Gathering options for training session.')
            training_cycles = trange(10)
            iters = trange(int(len(data) / 240))


            for i in training_cycles:
                tqdm.write("Round: {}".format(i))
                for iter_step in iters:
                    tqdm.write("Iter step {}".format(iter_step))
                    iter_data = data[
                                iter_step * 240: iter_step * 240 + 240
                                ]
                    # time.sleep(3)
                    # sys.exit()

                    # exit()
                    # build data set and reset to 0
                    env = StockEnv(iter_data)
                    s = env.reset()

                    # until break
                    while True:
                        # first S is data 0 from first dataloop
                        # s is recycled from bottom of loop
                        action = Agent.egreedy_action(s)

                        s_, reward, done = env.gostep(action)
                        # print ("Action: {} | 0 = buy, 1 = sell".format(action))
                        # print ("Reward: {:.4f}".format(reward))
                        # print ("Current Price set: {}".format(s_))
                        Agent.precive(s, action, reward, s_, done)
                        # print ("Prediction made...")
                        # print("Total Cash on hand?: {}".format(env.cash))
                        s = s_
                        if done:
                            # print("done")
                            tqdm.write("Total Cash on hand?: {}".format(env.cash))
                            break
                    # break
                # break
                Agent.save_model(step=i)
        # finish stats
        self.P('Total Runtime is {:.1f} mins'.format(float(timer()-runtime)/60))
        return True

    def get_dataset(self, exchange, pair):
        """Main Program."""
        coin1, coin2 = pair.split('/')
        df_filename = 'df_{}_{}_{}.csv'.format(exchange, coin1, coin2)
        ss_filename = 'ss_{}_{}_{}.csv'.format(exchange, coin1, coin2)
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # search for previous save files
        if self.options.save_file:
            if os.path.exists(os.path.join(dir_path, 'data', 'datasets', ss_filename)):
                self.P('Loading Saved Superset')
                ss = pd.read_csv(os.path.join(dir_path, 'data', 'datasets', ss_filename))
                self.P('Found {} elements per line'.format(len(ss.loc[0])))
            else:
                self.P('Fetching new Pricedata... This could take up to 21 mins per coin')
                df = self.dataHandler.get_candles(exchange, pair)
                # trunkating df size!!!!
                # df = df[:500]
                df.to_csv(os.path.join(dir_path, 'data', 'datasets', df_filename), index=False, header=True)
                self.P('Creating Superset.')
                start = timer()
                ss = df.superset()
                self.P('Took {:.1f} mins to build superset.'.format(float(timer() - start) / 60))
                ss.to_csv(os.path.join(dir_path, 'data', 'datasets', ss_filename), index=False, header=False)
                if True:
                    self.P('Fixing time in data')
                    start = timer()
                    ft = df.fix_time()
                    self.P('Took {:.1f} mins to build fixed time set.'.format(float(timer() - start) / 60))
                    self.plotter.Plot_me(ft, to_file=True)

        self.P('We have a superset, with a len of {}'.format(len(ss)))
        ss = ss.as_matrix()
        return ss


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
    app = Bittensor(config)
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
        sys.exit(e)
