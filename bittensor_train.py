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
        print('Finished Setup. Starting Training')
        # for i in range(2):  # RUN FOREVER!!!
        while True:
            sample_file = self.datasmith.random_filename
            print('Training file {}'.format(sample_file))
            self.datasmith.dataframe = sample_file
            df = self.datasmith.dataframe

            # figure out how many iters we can get out of this coin before
            # we run out of samples... the longer you have been saving the
            # data... obviously... the more you will have to train on.
            iters = len(df) - self.sample_size
            last_input = None
            inputs = None
            self.model.reset_que()
            for i in trange(iters - 1):  # dont run out of room
                iter_set = df[i:i+self.sample_size]
                normal = self.datasmith.make_normal(iter_set)
                last_input = inputs
                inputs = self.datasmith.make_input_from_normal(normal)
                if i == 0:
                    continue;
                action = self.model.egreedy_action(last_input)
                reward = self.model.get_reward(action, normal[1:-1])
                self.model.train(last_input, action, reward, inputs)

                # if i % 50 == 0:
                #     tqdm.write('this is working!!')
            self.model.save_or_load()
            print('Recap: steps {} | loss {}'.format(
                self.model.stats.g_step, self.model.stats.cost
            ))
        return True


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
