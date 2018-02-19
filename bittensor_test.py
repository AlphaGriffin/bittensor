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

# generic
import os, sys, time, datetime, collections, re, random, asyncio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from timeit import default_timer as timer
runtime = timer()
from tqdm import tqdm, trange
import numpy as np
import pandas as pd

import ag.bittensor.utils.options as options
import ag.bittensor.ai.make_data as make_data
from ag.bittensor.ai.AI import Q_Trader


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
        self.reset()

    def reset(self):
        # training Options
        self.sample_size = 80

        # playback options
        self.game_score = 0
        self.max_buys = 3
        self.curr_buys = 0

        self.total_buys = 0
        self.total_sells = 0
        self.do_nothing = 0

        self.target_buys = self.max_buys * 7
        self.starting_btc = 1
        self._buy_amount_btc = 0
        self.curr_btc = self.starting_btc
        self.curr_coin = 0
        self.total_profitable_moves = 0

        self.buys = []

    def main(self):
        print('Starting BitTensor')
        sample_file = self.datasmith.random_filename
        #print('getting file {}'.format(sample_file))
        self.datasmith.dataframe = sample_file
        df = self.datasmith.dataframe
        #print(self.datasmith.dataframe.tail(2))
        normal = self.datasmith.make_normal(df[0:self.sample_size])
        inputs = self.datasmith.make_input_from_normal(normal)
        #for i in inputs:
        #    print(len(i), i.shape)

        self.model.set_state_dim(
            inputs[0].shape[0],
            inputs[1].shape[0],
            inputs[2].shape[0],
            inputs[3].shape[0],
        )
        self.model.preRun()
        print('Finished Setup. Starting Testing')
        # for i in range(2):  # RUN FOREVER!!!
        while True:

            # sample_file = self.datasmith.next_filename
            sample_file = self.datasmith.random_filename
            print('Testing file {}'.format(sample_file))
            self.datasmith.dataframe = sample_file
            df = self.datasmith.dataframe

            # figure out how many iters we can get out of this coin before
            # we run out of samples... the longer you have been saving the
            # data... obviously... the more you will have to train on.
            iters = len(df) - self.sample_size
            self.reset()
            for i in trange(iters - 1):  # dont run out of room
                try:
                    if self.total_sells >= self.target_buys:
                        break
                    iter_set = df[i:i+self.sample_size]
                    normal = self.datasmith.make_normal(iter_set)
                    inputs = self.datasmith.make_input_from_normal(normal)
                    action = self.model.egreedy_action(inputs)
                    self.game(iter_set, action)
                except:
                    pass
            self.game(iter_set = iter_set, final=True)
            print('Pair: {}'.format(self.datasmith.pair))
            print('Total BTC after Trading: {:.8f}'.format(self.curr_btc))
            print('Total Performance: {:.2f}%'.format(self.total_profitable_moves/self.target_buys))
            print('Total BTC P/L after Trading: {:.2f}%'.format((self.curr_btc/self.starting_btc-1)*100))
            print('Total Buys/Sells/nothings: {}/{}/{}'.format(self.total_buys, self.total_sells, self.do_nothing))

    def game(self, iter_set, action=None, final=False):
        price = iter_set['last'].iloc[-1]
        if final:
            if self.buys:
                print('FUUUUCCCCCCKKKK!!!!!!!!')
                self.game_score -= self.curr_buys*price
            return
        if action == 0 and self.total_buys < self.target_buys:  # buy
            if len(self.buys) < self.max_buys:
                cost = float(self.curr_btc/self.max_buys)
                amount_to_buy =  cost / price
                self.curr_coin += amount_to_buy
                self.curr_btc -= cost
                self.buys.append((price, amount_to_buy))
                self.total_buys += 1
        elif action == 1:  # sell
            if self.buys:
                buy = min(self.buys)
                if price > buy[0]:
                    buy = max(self.buys)
                    self.total_profitable_moves += 1
                self.buys.remove(buy)
                self.curr_coin -= buy[1]
                rev = buy[1] * price
                self.curr_btc += rev
                self.total_sells += 1
        else:
            self.do_nothing += 1


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
