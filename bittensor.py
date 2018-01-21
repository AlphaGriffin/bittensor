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

# crypto
import ccxt

# Tensorflow
#from ag.bittensor.ai.stock_env import StockEnv
#from ag.bittensor.ai.DQN_trade import DQN_Trade as bot

# Utilities
import ag.bittensor.utils.options as options
import ag.bittensor.utils.plotter as plotter
import ag.bittensor.utils.sheets as sheets
import ag.bittensor.utils.printer as printer
import ag.logging as log

# set globals
log.set(log.DEBUG)


class Bittensor(object):
    """
    Bittensor.
    Another AlphaGriffin Project 2018.
    Alphagriffin.com
    """

    def __init__(self, options):
        """Use the options for a proper setup."""
        self.options = options

        # build objects
        self.plotter = plotter.Plot(options)
        self.sheets = sheets.gHooks(options)
        self.P = printer.Printer(options)
        #self.agent = bot()

        # CONSOLE LOGGING
        self.P('Setup Complete')
        log.debug('setup compete')

    def main(self):
        """sanity check."""
        if self.options:
            return True
        return False


def main():
    """Launcher for the app."""
    config = options.Options('config/dummy_codes.yaml')
    app = Bittensor(config)
    if app.main():
        return True
    return False

if __name__ == '__main__':
    try:
        import ag.bittensor.utils.options as options
        # os.system("mode con cols=80 lines=75")
        # os.system("clear")
        if main():
            print("AlphaGriffin  |  2018")
        else:
            print("Controlled Errors. Good day.")
    except Exception as e:
        print("and thats okay too.")
        sys.exit(e)
