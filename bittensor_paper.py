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

        # training Options
        self.sample_size = 80

        # playback options
        self.max_buys = 10
        self.total_buys = 0
        self.total_sells = 0
        self.do_nothing = 0
        self.target_buys = self.max_buys
        self.starting_btc = 1
        self.curr_btc = self.starting_btc
        self.curr_coin = {}
        self.total_profitable_moves = 0
        self._trade_history = {}


    @property
    def trade_history(self):
        return self._trade_history

    @trade_history.setter
    def trade_history(self, values):
        buy_data = values[0]
        curr_time = values[1]
        price = values[2]

        self.trade_history[curr_time] = dict()
        trade = self.trade_history[curr_time]
        trade['time'] = int(curr_time - buy_data['time'])  # change this to something useful
        trade['profit'] = float('{:.8f}'.format(price - buy_data['price'] * buy_data['volume']))
        trade['coin'] = buy_data['coin']

        pass

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
        print('Finished Setup in {:.2f} secs. Starting Testing'.format(timer()-runtime))
        while True:
            start_loop_time = None
            sample_file = self.datasmith.next_filename
            if sample_file is None:
                try:
                    loop_took = timer() - start_loop_time
                except:
                    loop_took = 1
                print('{} | loop time {:.2f}secs'.format(datetime.datetime.now(), loop_took))
                print(' :: Current Balances ::')
                btc_value = []
                for i in self.curr_coin:
                    btc_value.append(self.curr_coin[i]['cost'])
                    print('{} | {} @ {:.8f}'.format(i, self.curr_coin[i]['volume'], self.curr_coin[i]['price']))
                btc_value = np.array(btc_value).sum()
                print('BTC: {:.8f}'.format(self.curr_btc))
                print('Total Account Value in BTC: {:.8f}'.format(self.curr_btc + btc_value))
                print('Total Wins/Buys/Sells/nothings: {}/{}/{}/{}'.format(self.total_profitable_moves,
                                                                          self.total_buys,
                                                                          self.total_sells,
                                                                          self.do_nothing))
                start_loop_time = timer()
                print('Got to the end of the loop... Give it a min.')
                time.sleep(60)
                continue

            if 'BTC' not in sample_file.split('_')[1]:
                continue  # skip this one.
            self.datasmith.dataframe = sample_file
            df = self.datasmith.dataframe
            iter_set = df[len(df)-self.sample_size:len(df)]
            normal = self.datasmith.make_normal(iter_set)
            inputs = self.datasmith.make_input_from_normal(normal)
            action = self.model.egreedy_action(inputs)
            self.game(normal, action)

    def game(self, normal, action):
        msg = ' :: Game Engine Printout :: \n'
        price = normal[0]['last'].iloc[-1]
        coin = self.datasmith.pair.split('_')[0]
        msg += ' :: LOOKING AT {} | PAIR: {} :: \n'.format(coin, self.datasmith.pair)
        seven       = normal[1]['last'].iloc[-1]
        twelve      = normal[2]['last'].iloc[-1]
        twenty      = normal[3]['last'].iloc[-1]
        forty       = normal[4]['last'].iloc[-1]
        curr_time   = normal[5].iloc[-1]
        # action_ = 'BUY' if action == 0 else 'SELL'
        # msg += ' :: {} | Action: {} | Price: {:.8f} :: \n'.format(curr_time, action_, price)
        msg += ':: CHANGE ::\n'
        msg += '# 7 min change: {:.2f}\n'.format(seven)
        msg += '# 12 min change: {:.2f}\n'.format(twelve)
        msg += '# 20 min change: {:.2f}\n'.format(twenty)
        msg += '# 40 min change: {:.2f}\n'.format(forty)

        if action == 0 and len(self.curr_coin) < self.target_buys:  # if buy
            if len(self.curr_coin) < self.max_buys:  # and there are buys left
                cost = float(self.curr_btc/self.max_buys)
                self.total_buys += 1
                amount_to_buy =  cost / price
                self.curr_coin[coin] = dict()
                self.curr_coin[coin]['coin'] = coin
                self.curr_coin[coin]['price'] = float('{:.8f}'.format(price))
                self.curr_coin[coin]['volume'] = float('{:.8f}'.format(amount_to_buy))
                self.curr_coin[coin]['cost'] = float('{:.8f}'.format(cost))
                self.curr_coin[coin]['time'] = curr_time
                self.curr_btc -= cost
                msg += ' :: Time of Action: {} ::\n'.format(curr_time)
                msg += ' :: Buying {:.4f} of {} @ {:.8f} | cost BTC: {:.8f} ::\n'.format(amount_to_buy, coin, price, cost)
                print(msg)

        elif action == 1:  # if sell
            if coin in self.curr_coin:  # and if we have this coin
                buy_data = self.curr_coin[coin]
                if float(price) > float(buy_data['price']):
                    self.total_profitable_moves += 1
                del self.curr_coin[coin]
                self.total_sells += 1
                rev = buy_data['volume'] * price
                self.curr_btc += rev
                self.total_sells += 1
                msg += ' :: Time of Action: {} ::\n'.format(curr_time)
                msg += ' :: Selling {:.4f} of {} @ {:.8f} | revenue BTC: {:.8f} ::\n'.format(buy_data['volume'], coin, price, rev)
                print(msg)
                self.trade_history = [buy_data, curr_time, price]

        if action > 1:
            self.do_nothing += 1

        # return msg

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
