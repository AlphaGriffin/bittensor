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
from collections import deque

# Tensorflow
import tensorflow as tf
import pandas as pd
import numpy as np
import random
#from stock_env import StockEnv
#from ag.bittensor.ai.stock_env import StockEnv

# Dataset
import ag.bittensor.ai.dataprep as dataprep

# GameEngine // need access to engine for reward data
import ag.bittensor.game.engine as engine


class DQN_Trader(object):
    """
    This handles all the flowing tensors.
    """

    def __init__(self, options):
        self.options = options
        self.dataHandler = dataprep.DataHandler()
        self.gameengine = engine.GameEngine(options)
