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

#//////////////////////////////////////////////////////////////////////////////
# This is a test of the system of what it would do in real time in actual game
# play mode.

# Build Globals
config = options.Options('config/access_codes.yaml')
pprint = printer.Printer(config)
robot = bot(config)

pprint('AlphaGriffin | 2018')
